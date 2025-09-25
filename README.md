# RL makes MLLMs see better than SFT

Official PyTorch implementation of ICLR 2026 Submission #3597 (anonymous)


## 1. Install

```bash
# bash
git clone https://github.com/anonymous-1125/RLmakesMLLMsSeeBetter.git
cd RLmakesMLLMsSeeBetter
conda create -n llava python=3.10.12 -y
conda activate llava
pip install --upgrade pip  # enables PEP 660 support for editable installs
pip install -e ".[train]"
```

### Docker (optional)

Recommended base image: `nvcr.io/nvidia/pytorch:23.04-py3`
Note: if you use this container, remove any torch/torchvision entries from pyproject.toml to avoid version conflicts.

## 2. Dataset

| Stage                             | Description                     | Source                                                                                                                                                                                             |
| --------------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stage1 Pretraining projector-only | BLIP/LAION/CC/SBU 558k manifest | [https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json) |
| Stage1 Pretraining full training  | LLaVA-OneVision-Data            | [https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)                                                                     |
| Stage2 Post-training              | MMPR-v1.2                       | [https://huggingface.co/datasets/OpenGVLab/MMPR-v1.2](https://huggingface.co/datasets/OpenGVLab/MMPR-v1.2)                                                                                         |

## 3. Train

Our training process consists of two main stages as described in the paper.

### Stage1: Pre-training

```bash
# bash
bash scripts/train/stage1-proj-train.sh
bash scripts/train/stage1-full-train.sh
```

### Stage2: Post-training


```bash
# bash
bash scripts/train/stage2-sft.sh
# or
bash scripts/train/stage2-dpo.sh
```

## 4. Eval

We provide comprehensive evaluation scripts, covering both MLLM benchmarks and specific analyses of the vision encoder's representations.

### 4.1 (MLLM) Cambrian

#### Setup
```bash
# bash
pip install mmh3
sudo apt-get install -y libwebp-dev libjpeg-dev zlib1g-dev
pip install --upgrade "Pillow[webp]>=10.0.0"
pip install nltk
pip install openpyxl
```

#### Run
```bash
# bash
bash scripts/run_all_benchmarks.sh <Path to MLLM checkpoint root>
python scripts/tabulate.py  # aggregate results into a table
```

For more details on the evaluation, please refer to the official repository: https://github.com/cambrian-mllm/cambrian/tree/main/eval

### 4.2 (MLLM) lams-eval

#### Install
```bash
# bash
cd RLmakesMLLMsSeeBetter/lmms-eval
pip install -e .
pip install httpx==0.23.3
pip install protobuf==3.20
pip install langdetect
pip install immutabledict
```

```python
# bash
python 
# python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')  # key
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

#### Run

```bash
# bash
python -m accelerate.commands.launch \
  --num_processes=1 \   
  --main_process_port 20006 \
  -m lmms_eval \
  --model_args pretrained=${ROOT_PATH}/${MODEL_PRETRAINED},conv_template=qwen_1_5,model_name=llava_qwen,attn_implementation=sdpa \
  --model llava_onevision \
  --tasks docvqa_val \
  --batch_size 1 \
  --log_samples_suffix llava_onevision
```

For more examples and model configs, please refer to the official repository: https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/examples/models


### 4.3 (Vision encoder) ImageNet

```python
# bash
cd eval_vit
python imagenet_vit_projector_prototype.py \
  --pretrained <path-to-mllm-checkpoint-root> \
  --imagenet_root ./imagenet/ILSVRC2012 \
  --batch_size 64 --workers 8 --subset_per_class 50
```


### 4.4 (Vision encoder) GradCam

To enable Grad-CAM visualization, add the `--draw_gradcam True` flag to the `scripts/train/stage2-sft.sh` and `scripts/train/stage2-dpo.sh` scripts.


### 4.5 For more custom analysis with the vision encoder

```python
import torch
from PIL import Image
from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor
from eval_vit.utils import from_path_to_vision_encoder, build_projector

vision_base_arch = "google/siglip2-so400m-patch16-384"
weight_dir = "./checkpoints/stage1-full-train/stage2-sft-siglip2-so500m-Qwen2.5-1.5B-llava" # <Path to MLLM checkpoint root>
vision_weight = from_path_to_vision_encoder(weight_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

config = SiglipVisionConfig.from_pretrained(vision_base_arch)
vit = SiglipVisionModel.from_pretrained(vision_base_arch)
processor = SiglipImageProcessor.from_pretrained(vision_base_arch)

del vit.vision_model.encoder.layers[-1:]    
vit.vision_model.head = torch.nn.Identity()
vit.eval().to(device)

vit_sd = vision_weight["vision_tower"]  
vit_sd = {k.replace("vision_tower.", "", 1): v for k, v in vit_sd.items()}
missing, unexpected = vit.load_state_dict(vit_sd, strict=True)

proj_sd = vision_weight["mm_projector"] 
projector = build_projector(proj_sd, device)
missing_p, unexpected_p = projector.load_state_dict(proj_sd, strict=True)

IMAGE_PATH = "dog.jpg"

img = Image.open(IMAGE_PATH).convert("RGB")
batch = processor.preprocess(img, return_tensors="pt")  
pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32)

with torch.no_grad():
    vit_out = vit(pixel_values=pixel_values, output_hidden_states=True)
    vit_feats = vit_out.hidden_states[-1]           # [batch, num_patchs, vision_hidden_dim]
    proj_feats = projector(vit_feats)               # [batch, num_patchs, llm_hidden_dim]

print("vit feature shape:", tuple(vit_feats.shape))
print("projected feature shape:", tuple(proj_feats.shape))
```

# Acknowledgements

Our work builds upon several open-source projects:

- [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [CLIP](https://github.com/openai/CLIP)
- [Cambrian](https://github.com/cambrian-mllm/cambrian)
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)