# Few-shot LoRA Training System (Docker Compose)

A docker container for 

Highlights:
- **Reproducible**: everything runs inside a container (no local Python env needed).
- **Simple**: one command to (optionally) caption images + train.
- **Safe defaults** for few-shot SDXL LoRA.
- **Includes inference**: SDXL txt2img with LoRA using `diffusers`.

---

## Requirements

- Docker + Docker Compose v2
- A GPU-enabled Docker runtime is strongly recommended for training.

---

## Build

```bash
docker compose build trainer
```
---

## Train (caption + LoRA)

```bash
# train
docker compose run --rm trainer train \
--base-model /models/base/sd_xl_base_1.0.safetensors \
--images /datasets/subject/images \
--run-name title \
--sdxl \
--caption-mode blip \
--concept-token sksSubject \
--max-train-steps 1600 \
--num-repeats 20 \
--network-dim 16 \
--network-alpha 8
```

## Caption only (BLIP)

If you want to generate `.txt` captions next to each image (same basename):

```bash
# ## Caption only (BLIP)
docker compose run 
--rm trainer caption   
--images /datasets/subject/images   
--prefix sksSubject   
--overwrite
```

## Inference (SDXL txt2img with LoRA)

Generate images with the trained LoRA:

```bash
# inference
docker compose run 
--rm trainer infer   
--base-model /models/base/sd_xl_base_1.0.safetensors   
--lora /models/loras/subject_***.safetensors   
--prompt "sksSubject portrait photo"   
--negative-prompt ""   
--out-dir /work/outputs   
--num-images 4   
--steps 30   
--cfg 7.0   
--width 1024   
--height 1024   
--lora-scale 0.8   
--seed 42
```
---

## License
- Apache License 2.0