# [SDXL Container](https://github.com/europanite/sdxl_container "SDXL Container")

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11|%203.12|%203.13-blue)](https://www.python.org/)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)

[![CI](https://github.com/europanite/sdxl_container/actions/workflows/ci.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/ci.yml)
[![CodeQL Advanced](https://github.com/europanite/sdxl_container/actions/workflows/codeql.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/codeql.yml)
[![Pytest](https://github.com/europanite/sdxl_container/actions/workflows/pytest.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/pytest.yml)
[![Python Lint](https://github.com/europanite/sdxl_container/actions/workflows/lint.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/lint.yml)
[![pages](https://github.com/europanite/sdxl_container/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/pages/pages-build-deployment)


<p align="right">
  <a href="./README.md">🇺🇸 English</a> |
  <a href="./README.hi.md">🇮🇳 हिंदी</a> |
  <a href="./README.ja.md">🇯🇵 日本語</a> |
  <a href="./README.zh-CN.md">🇨🇳 简体中文</a> |
  <a href="./README.es.md">🇪🇸 Español</a> |
  <a href="./README.pt-BR.md">🇧🇷 Português (Brasil)</a> |
  <a href="./README.ko.md">🇰🇷 한국어</a> |
  <a href="./README.de.md">🇩🇪 Deutsch</a> |
  <a href="./README.fr.md">🇫🇷 Français</a>
</p>


!["image"](./assets/images/image.png)

**SDXL LoRA adapters를 train**하고 **SDXL inference를 run**하기 위한 docker container입니다.

이 repo는 “small image set” LoRA runs에 최적화되어 있습니다:
1) images를 folder에 넣습니다  
2) (선택 사항) captions를 auto-generate합니다  
3) LoRA를 `./models/loras/`에 train합니다  
4) 해당 LoRA로 즉시 images를 generate합니다

---

## What’s inside

- **GPU trainer container**
- **Command entrypoint**: `train` / `caption` / `infer`
- **LoRA training wrapper** 
- **Training launcher wrapper**
- **BLIP captioning tool**
- **Diffusers inference script**
- CI용 **CPU-only test container**

---

## Architecture / Mounts

`docker-compose.yml`은 local folders를 container 안에 mount합니다:

- `./models`   → `/models`   (base models + output LoRAs)
- `./datasets` → `/datasets` (your raw images)
- `./workspace`→ `/workspace`(runs + caches + outputs)
- `./scripts`  → `/scripts`  (entrypoint + wrappers)

모든 commands는 container 내부에서 실행되지만, files는 이러한 mounts를 통해 host에 기록됩니다.

---

## Prerequisites

- Docker + Docker Compose
- GPU + toolkit (`gpus: all`용)
- SDXL base model. 다음 중 하나: (a) `./models/base/` 아래의 local `.safetensors`/diffusers dir, 또는 (b) Hugging Face repo id (예: `stabilityai/sdxl-turbo`)
 - `./datasets/<subject>/images/` 아래의 small dataset
---

Highlights:
- **Reproducible**: 모든 것이 container 안에서 실행됩니다 (local Python env가 필요하지 않음).
- **Simple**: images caption 생성(선택 사항) + train을 하나의 command로 실행합니다.
- few-shot SDXL LoRA를 위한 **Safe defaults**.
- **Includes inference**: `diffusers`를 사용한 LoRA 포함 SDXL txt2img.

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
--base-model stabilityai/sdxl-turbo \
--images /datasets/yokosuka \
--run-name yokosuka \
--sdxl \
--caption-mode blip \
--concept-token sksyokosuka \
--max-train-steps 1600 \
--num-repeats 20 \
--network-dim 16 \
--network-alpha 8
```

## Infer (txt2img)
```bash
docker compose run --rm trainer infer \
--base-model stabilityai/sdxl-turbo \
--lora /models/loras/title_***.safetensors \
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

각 image 옆에 같은 basename의 `.txt` captions를 generate하려는 경우:

```bash
# caption
docker compose run  \
--rm trainer caption  \  
--images /datasets/title \   
--prefix sksSubject    \
--overwrite
```

## Inference (SDXL txt2img with LoRA)

Trained LoRA로 images를 generate합니다:

```bash
# inference
docker compose run  \
--rm trainer infer    \
--base-model /models/base/sd_xl_base_1.0.safetensors    \
--lora /models/loras/title_***.safetensors    \
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

##  LoRA algorithm 

LoRA (Low-Rank Adaptation)는 base weights를 frozen 상태로 유지하면서 선택된 weight matrices에 low-rank update를 추가하여 diffusion model을 fine-tune합니다.

weight matrix W에 대해 LoRA는 다음을 학습합니다:

ΔW = (α / r) * (B @ A)

여기서:

r은 rank입니다 (--network-dim)

α는 scaling factor입니다 (--network-alpha)

A와 B는 low-rank trainable matrices입니다

Inference time에는 effective weight가 다음과 같이 됩니다:

W' = W + ΔW

또한 이 repo에서는 --lora-scale을 통해 LoRA가 generation에 얼마나 강하게 영향을 주는지 제어할 수 있습니다.

## License
- Apache License 2.0
