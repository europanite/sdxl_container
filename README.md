# [SDXL Container](https://github.com/europanite/sdxl_container "SDXL Container")

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11|%203.12|%203.13-blue)](https://www.python.org/)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)

[![CI](https://github.com/europanite/sdxl_container/actions/workflows/ci.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/ci.yml)
[![CodeQL Advanced](https://github.com/europanite/sdxl_container/actions/workflows/codeql.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/codeql.yml)
[![Pytest](https://github.com/europanite/sdxl_container/actions/workflows/pytest.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/pytest.yml)
[![Python Lint](https://github.com/europanite/sdxl_container/actions/workflows/lint.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/lint.yml)
[![pages](https://github.com/europanite/sdxl_container/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/pages/pages-build-deployment)

A docker container for SDXL

!["image"](./assets/images/image.png)

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

## Caption (BLIP)

If you want to generate `.txt` captions next to each image (same basename):

```bash
# caption
docker compose run  \
--rm trainer caption  \  
--images /datasets/subject \   
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

## License
- Apache License 2.0