# [SDXL Container](https://github.com/europanite/sdxl_container "SDXL Container")

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11|%203.12|%203.13-blue)](https://www.python.org/)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)

[![CI](https://github.com/europanite/sdxl_container/actions/workflows/ci.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/ci.yml)
[![CodeQL Advanced](https://github.com/europanite/sdxl_container/actions/workflows/codeql.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/codeql.yml)
[![Pytest](https://github.com/europanite/sdxl_container/actions/workflows/pytest.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/pytest.yml)
[![Python Lint](https://github.com/europanite/sdxl_container/actions/workflows/lint.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/lint.yml)
[![pages](https://github.com/europanite/sdxl_container/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/pages/pages-build-deployment)


!["image"](./assets/images/image.png)

A docker container to **train SDXL LoRA adapters** and **run SDXL inference**.

This repo is optimized for “small image set” LoRA runs:
1) drop images into a folder  
2) (optionally) auto-generate captions  
3) train a LoRA into `./models/loras/`  
4) immediately generate images with that LoRA

---

## What’s inside

- **GPU trainer container**
- **Command entrypoint**: `train` / `caption` / `infer`
- **LoRA training wrapper** 
- **Training launcher wrapper**
- **BLIP captioning tool**
- **Diffusers inference script**
- **CPU-only test container** for CI

---

## Architecture / Mounts

`docker-compose.yml` mounts local folders into the container:

- `./models`   → `/models`   (base models + output LoRAs)
- `./datasets` → `/datasets` (your raw images)
- `./workspace`→ `/workspace`(runs + caches + outputs)
- `./scripts`  → `/scripts`  (entrypoint + wrappers)

All commands run inside the container, but files are written to your host via these mounts.

---

## Prerequisites

- Docker + Docker Compose
- GPU + toolkit (for `gpus: all`)
- An SDXL base model you are allowed to use (e.g. `sd_xl_base_1.0.safetensors`)

---

Highlights:
- **Reproducible**: everything runs inside a container (no local Python env needed).
- **Simple**: one command to (optionally) caption images + train.
- **Safe defaults** for few-shot SDXL LoRA.
- **Includes inference**: SDXL txt2img with LoRA using `diffusers`.

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
--images /datasets/title \
--run-name title \
--sdxl \
--caption-mode blip \
--concept-token sksSubject \
--max-train-steps 1600 \
--num-repeats 20 \
--network-dim 16 \
--network-alpha 8
```

## Infer (txt2img)
```bash
docker compose run --rm trainer infer \
--base-model /models/base/sd_xl_base_1.0.safetensors \
--lora /models/loras/title_20260207_123456.safetensors \
--prompt "portrait photo of sksTitle, high detail, natural light" \
--negative-prompt "low quality, blurry, worst quality" \
--out-dir /workspace/outputs \
--num-images 4 \
--seed 123 \
--steps 30 \
--cfg 7.0 \
--lora-scale 0.8 \
--width 1024 --height 1024
```

## Caption (BLIP)

If you want to generate `.txt` captions next to each image (same basename):

```bash
# caption
docker compose run  \
--rm trainer caption  \  
--images /datasets/title \   
--prefix sksSubject    \
--overwrite
```

## Inference (SDXL txt2img with LoRA)

Generate images with the trained LoRA:

```bash
# inference
docker compose run  \
--rm trainer infer    \
--base-model /models/base/sd_xl_base_1.0.safetensors    \
--lora /models/loras/title_20260204_123246.safetensors    \
--prompt "sksSubject seaside"    \
--negative-prompt ""    \
--out-dir /datasets/title/inference    \
--num-images 4    \
--steps 30    \
--cfg 7.0    \
--width 1024    \
--height 1024    \
--lora-scale 0.8    \
--seed 42
```
---

## Test
```bash
docker compose -f docker-compose.test.yml build
docker compose -f docker-compose.test.yml run --rm test
```


LoRA algorithm (what is being learned)

LoRA (Low-Rank Adaptation) fine-tunes a diffusion model by adding a low-rank update to selected weight matrices while keeping the base weights frozen.

For a weight matrix W, LoRA learns:

ΔW = (α / r) * (B @ A)

Where:

r is the rank (--network-dim)

α is the scaling factor (--network-alpha)

A and B are the low-rank trainable matrices

At inference time the effective weight becomes:

W' = W + ΔW

Additionally, this repo lets you control how strongly the LoRA influences generation via --lora-scale.

## License
- Apache License 2.0