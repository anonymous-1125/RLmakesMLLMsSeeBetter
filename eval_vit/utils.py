from pathlib import Path
import torch.nn as nn
from safetensors import safe_open


def from_path_to_vision_encoder(path):
    want_prefixes = [
        "model.mm_projector.", "mm_projector.",
        "model.vision_tower.", "vision_tower."
    ]

    vit_sd, proj_sd = {}, {}

    def take(key, tensor):
        clean = key
        for p in want_prefixes:
            if clean.startswith(p):
                clean = clean[len(p):]
                break
        if key.split(".")[1] == "mm_projector" or key.startswith("mm_projector."):
            proj_sd[clean] = tensor.cpu()
        else:
            vit_sd[clean] = tensor.cpu()

    for p in Path(path).glob("**/*.safetensors"):
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                if any(k.startswith(pref) for pref in want_prefixes):
                    take(k, f.get_tensor(k))

    pack = {
        "vision_tower": vit_sd,
        "mm_projector": proj_sd,
        "meta": {
            "pretrained": path,
            "method": "partial-load-from-safetensors",
        },
    }

    return pack


def build_projector(projector_weight, device):
    w0 = projector_weight["0.weight"]  
    w2 = projector_weight["2.weight"]  
    mm_hidden_size = w0.shape[1]
    hidden_size    = w0.shape[0]

    projector = nn.Sequential(
                    nn.Linear(mm_hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                ).to(device).eval()
    
    return projector