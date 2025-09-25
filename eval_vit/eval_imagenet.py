import os
import csv
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from tqdm import tqdm

from PIL import Image
from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor
from utils import from_path_to_vision_encoder, build_projector
from sklearn.linear_model import LogisticRegression


# ------------------------------
# Small helpers
# ------------------------------

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def build_imagenet_loader(root, split, processor, batch_size=64, workers=8, indices=None):
    assert split in {"train", "val"}

    def transform(img: Image.Image):
        feat = processor.preprocess(img, return_tensors="pt")
        return feat["pixel_values"][0]

    data_path = os.path.join(root, split)
    dataset = datasets.ImageFolder(data_path, transform)
    sampler = SubsetRandomSampler(indices) if indices is not None else None
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, sampler=sampler, shuffle=False), dataset


def balanced_indices(dataset, num_per_class=50, seed=0):
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    indices = []
    for c in range(max(targets) + 1):
        cls_idxs = np.where(targets == c)[0]
        if len(cls_idxs) < num_per_class:
            raise ValueError(f"Class {c} has only {len(cls_idxs)} samples < {num_per_class}")
        pick = rng.choice(cls_idxs, size=num_per_class, replace=False)
        indices.append(pick)
    return np.concatenate(indices)


essential_Cs = [0.01, 0.03, 0.1, 0.316, 1.0, 3.16]


def extract_features(dataloader, feat_fn, device, dtype):
    feats, labels = [], []
    with torch.no_grad():
        for images, y in tqdm(dataloader, desc="Extract"):
            images = images.to(device=device, dtype=dtype)
            patch = feat_fn(images)                    # [B, P, D]
            g = F.normalize(patch.mean(dim=1), dim=-1) # [B, D]
            feats.append(g.cpu())
            labels.append(y)
    X = torch.cat(feats).numpy()
    y = torch.cat(labels).numpy()
    return X, y


def train_logreg_grid(X_train, y_train, X_val, y_val, Cs=essential_Cs, max_iter=1000):
    best_acc, best_C, best_clf = -1.0, None, None
    for C in Cs:
        clf = LogisticRegression(
            random_state=0, C=C, max_iter=max_iter, verbose=0,
            multi_class="auto", n_jobs=None, solver="lbfgs"
        )
        clf.fit(X_train, y_train)
        acc = (clf.predict(X_val) == y_val).mean()
        if acc > best_acc:
            best_acc, best_C, best_clf = acc, C, clf
    return best_clf, best_acc, best_C


# ------------------------------
# Main
# ------------------------------

def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Faster/Fast/Slow evaluations for SigLIP ViT + projector")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to MLLM checkpoint root",\
                        default="./checkpoints/stage1-full-train/stage2-sft-siglip2-so500m-Qwen2.5-1.5B-llava")
    parser.add_argument("--imagenet_root", type=str, default="./ILSVRC2012")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--csv_out", type=str, default=os.path.join("eval_vit", "table_values.csv"))
    parser.add_argument("--subset_per_class", type=int, default=50, help="Fast eval: images per class (50 => 50K)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    weight_dir = args.pretrained
    vision_base_arch = "google/siglip2-so400m-patch16-384"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = Path(weight_dir.rstrip("/" )).name

    # ViT + projector from MLLM weights
    config = SiglipVisionConfig.from_pretrained(vision_base_arch)
    vit = SiglipVisionModel.from_pretrained(vision_base_arch)
    processor = SiglipImageProcessor.from_pretrained(vision_base_arch)
    del vit.vision_model.encoder.layers[-1:]
    vit.vision_model.head = torch.nn.Identity()
    vit.eval().to(device)

    vision_weight = from_path_to_vision_encoder(weight_dir)
    vit_sd = {k.replace("vision_tower.", "", 1): v for k, v in vision_weight["vision_tower"].items()}
    vit.load_state_dict(vit_sd, strict=True)

    proj_sd = vision_weight["mm_projector"]
    projector = build_projector(proj_sd, device)
    projector.load_state_dict(proj_sd, strict=True)
    projector.eval()

    # Feature fns
    model_dtype = next(vit.parameters()).dtype

    def feat_vit(images: torch.Tensor) -> torch.Tensor:
        out = vit(pixel_values=images, output_hidden_states=True)
        return out.hidden_states[-1]

    def feat_projected(images: torch.Tensor) -> torch.Tensor:
        return projector(feat_vit(images))

    # Data loaders
    val_loader, val_ds = build_imagenet_loader(args.imagenet_root, "val", processor, args.batch_size, args.workers)

    # ==========================
    # 1) Faster evaluation
    # ==========================
    print("\n===== Faster evaluation =====")

    def compute_prototypes(feat_fn, hidden_dim):
        sum_feats = torch.zeros(1000, hidden_dim, device=device, dtype=model_dtype)
        counts = torch.zeros(1000, device=device)
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Prototypes"):
                images = images.to(device=device, dtype=model_dtype)
                labels = labels.to(device)
                patch = feat_fn(images)
                g = F.normalize(patch.mean(dim=1), dim=-1)
                for feat, lbl in zip(g, labels):
                    sum_feats[lbl] += feat
                    counts[lbl] += 1
        return (sum_feats / counts.unsqueeze(1)).t()

    proj_hidden = projector(torch.zeros(1, 2, vit.config.hidden_size, device=device)).shape[-1]
    proto_proj = compute_prototypes(feat_projected, proj_hidden)
    vit_hidden = vit.config.hidden_size
    proto_vit = compute_prototypes(feat_vit, vit_hidden)

    def eval_with_classifier(feat_fn, classifier):
        top1 = top5 = n = 0.0
        with torch.no_grad():
            for images, target in tqdm(val_loader, unit="batch"):
                images = images.to(device=device, dtype=model_dtype)
                target = target.to(device)
                patch = feat_fn(images)
                img = F.normalize(patch.mean(dim=1), dim=-1)
                logits = 100.0 * img @ classifier.to(dtype=model_dtype)
                a1, a5 = accuracy(logits, target, topk=(1, 5))
                top1 += a1; top5 += a5; n += images.size(0)
        return top1 / n, top5 / n

    t1_p, t5_p = eval_with_classifier(feat_projected, proto_proj)
    t1_v, t5_v = eval_with_classifier(feat_vit, proto_vit)
    print(f"[Faster][ViT+Projector] Top-1={t1_p*100:.2f}%, Top-5={t5_p*100:.2f}%")
    print(f"[Faster][ViT-only]     Top-1={t1_v*100:.2f}%, Top-5={t5_v*100:.2f}%")

    # ==========================
    # 2) Fast evaluation (50K balanced linear probing)
    # ==========================
    print("\n===== Fast evaluation (50K balanced linear probing) =====")

    train_loader_full, train_ds = build_imagenet_loader(args.imagenet_root, "train", processor, args.batch_size, args.workers)
    idx_balanced = balanced_indices(train_ds, num_per_class=args.subset_per_class, seed=args.seed)
    train_loader_50k, _ = build_imagenet_loader(args.imagenet_root, "train", processor, args.batch_size, args.workers, indices=idx_balanced)

    Xtr_proj, ytr = extract_features(train_loader_50k, feat_projected, device, model_dtype)
    Xte_proj, yte = extract_features(val_loader,      feat_projected, device, model_dtype)
    clf_proj, acc_proj, bestC_proj = train_logreg_grid(Xtr_proj, ytr, Xte_proj, yte)
    print(f"[Fast][ViT+Projector] Val acc={acc_proj*100:.2f}% (best C={bestC_proj})")

    Xtr_vit, ytr_v = extract_features(train_loader_50k, feat_vit, device, model_dtype)
    Xte_vit, yte_v = extract_features(val_loader,      feat_vit, device, model_dtype)
    clf_vit, acc_vit, bestC_vit = train_logreg_grid(Xtr_vit, ytr_v, Xte_vit, yte_v)
    print(f"[Fast][ViT-only]      Val acc={acc_vit*100:.2f}% (best C={bestC_vit})")

    # ==========================
    # 3) Slow evaluation (full-train linear probing)
    # ==========================
    print("\n===== Slow evaluation (full-train linear probing) =====")

    Xtr_full_proj, ytr_full = extract_features(train_loader_full, feat_projected, device, model_dtype)
    clf_full_proj, acc_full_proj, bestC_full_proj = train_logreg_grid(Xtr_full_proj, ytr_full, Xte_proj, yte)
    print(f"[Slow][ViT+Projector] Val acc={acc_full_proj*100:.2f}% (best C={bestC_full_proj})")

    Xtr_full_vit, ytr_full_v = extract_features(train_loader_full, feat_vit, device, model_dtype)
    clf_full_vit, acc_full_vit, bestC_full_vit = train_logreg_grid(Xtr_full_vit, ytr_full_v, Xte_vit, yte_v)
    print(f"[Slow][ViT-only]      Val acc={acc_full_vit*100:.2f}% (best C={bestC_full_vit})")

    # CSV: model_name and metrics
    row = [
        model_name,
        f"{t1_p*100:.2f}", f"{t5_p*100:.2f}", f"{t1_v*100:.2f}", f"{t5_v*100:.2f}",
        f"{acc_proj*100:.2f}", f"{acc_vit*100:.2f}",
        f"{acc_full_proj*100:.2f}", f"{acc_full_vit*100:.2f}",
    ]
    with open(args.csv_out, mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
    print(f"Saved results to {args.csv_out}")


if __name__ == "__main__":
    main()
