---
layout: page
title: "🇯🇵 日本語"
permalink: /ja/
lang: ja
---

# [SDXL Container](https://github.com/europanite/sdxl_container "A Docker Container to train SDXL LoRA adapters and run SDXL inference")

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11|%203.12|%203.13-blue)](https://www.python.org/)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)

[![CI](https://github.com/europanite/sdxl_container/actions/workflows/ci.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/ci.yml)
[![CodeQL Advanced](https://github.com/europanite/sdxl_container/actions/workflows/codeql.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/codeql.yml)
[![Pytest](https://github.com/europanite/sdxl_container/actions/workflows/pytest.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/pytest.yml)
[![Python Lint](https://github.com/europanite/sdxl_container/actions/workflows/lint.yml/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/lint.yml)
[![pages](https://github.com/europanite/sdxl_container/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/europanite/sdxl_container/actions/workflows/pages/pages-build-deployment)


<p align="right">
  <a href="https://europanite.github.io/sdxl_container/">🇺🇸 English</a> |
  <a href="https://europanite.github.io/sdxl_container/hi/">🇮🇳 हिंदी</a> |
  <a href="https://europanite.github.io/sdxl_container/ja/">🇯🇵 日本語</a> |
  <a href="https://europanite.github.io/sdxl_container/zh-CN/">🇨🇳 简体中文</a> |
  <a href="https://europanite.github.io/sdxl_container/es/">🇪🇸 Español</a> |
  <a href="https://europanite.github.io/sdxl_container/pt-BR/">🇧🇷 Português (Brasil)</a> |
  <a href="https://europanite.github.io/sdxl_container/ko/">🇰🇷 한국어</a> |
  <a href="https://europanite.github.io/sdxl_container/de/">🇩🇪 Deutsch</a> |
  <a href="https://europanite.github.io/sdxl_container/fr/">🇫🇷 Français</a>
</p>


!["image"](./assets/images/image.png)

**SDXL LoRA adapters を train**し、**SDXL inference を実行**するための docker container。

この repo は “small image set” の LoRA runs 向けに最適化されています:
1) images を folder に入れる  
2) （任意で）captions を auto-generate する  
3) LoRA を `./models/loras/` に train する  
4) その LoRA ですぐに images を generate する

---

## What’s inside

- **GPU trainer container**
- **Command entrypoint**: `train` / `caption` / `infer`
- **LoRA training wrapper** 
- **Training launcher wrapper**
- **BLIP captioning tool**
- **Diffusers inference script**
- **CI 用の CPU-only test container**

---

## Architecture / Mounts

`docker-compose.yml` は local folders を container に mount します:

- `./models`   → `/models`   (base models + output LoRAs)
- `./datasets` → `/datasets` (your raw images)
- `./workspace`→ `/workspace`(runs + caches + outputs)
- `./scripts`  → `/scripts`  (entrypoint + wrappers)

すべての commands は container 内で実行されますが、files はこれらの mounts を通じて host に書き込まれます。

---

## Prerequisites

- Docker + Docker Compose
- GPU + toolkit (`gpus: all` 用)
- SDXL base model。次のいずれか: (a) `./models/base/` 配下の local `.safetensors`/diffusers dir、または (b) Hugging Face repo id（例: `stabilityai/sdxl-turbo`）
 - `./datasets/<subject>/images/` 配下の small dataset
---

Highlights:
- **Reproducible**: すべて container 内で実行されます（local Python env は不要）。
- **Simple**: images の caption 生成（任意）+ train を 1 command で実行できます。
- **Safe defaults** for few-shot SDXL LoRA。
- **Includes inference**: `diffusers` を使った LoRA 付き SDXL txt2img。

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

各 image の横に、同じ basename の `.txt` captions を generate したい場合:

```bash
# caption
docker compose run  \
--rm trainer caption  \  
--images /datasets/title \   
--prefix sksSubject    \
--overwrite
```

## Inference (SDXL txt2img with LoRA)

Trained LoRA で images を generate します:

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

LoRA (Low-Rank Adaptation) は、base weights を frozen のままにし、選択した weight matrices に low-rank update を追加することで diffusion model を fine-tune します。

weight matrix W に対して、LoRA は次を学習します:

ΔW = (α / r) * (B @ A)

ここで:

r は rank です (--network-dim)

α は scaling factor です (--network-alpha)

A と B は low-rank trainable matrices です

Inference time では effective weight は次のようになります:

W' = W + ΔW

さらに、この repo では --lora-scale によって、LoRA が generation にどれだけ強く影響するかを制御できます。

## License
- Apache License 2.0
