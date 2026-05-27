---
layout: page
title: "🇨🇳 简体中文"
permalink: /zh-CN/
lang: zh-CN
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

用于**训练 SDXL LoRA adapters**并**运行 SDXL inference**的 docker container。

此 repo 针对 “small image set” LoRA runs 进行了优化:
1) 将 images 放入一个 folder  
2) （可选）auto-generate captions  
3) 将 LoRA train 到 `./models/loras/`  
4) 立即使用该 LoRA generate images

---

## What’s inside

- **GPU trainer container**
- **Command entrypoint**: `train` / `caption` / `infer`
- **LoRA training wrapper** 
- **Training launcher wrapper**
- **BLIP captioning tool**
- **Diffusers inference script**
- **用于 CI 的 CPU-only test container**

---

## Architecture / Mounts

`docker-compose.yml` 将 local folders mount 到 container 中:

- `./models`   → `/models`   (base models + output LoRAs)
- `./datasets` → `/datasets` (your raw images)
- `./workspace`→ `/workspace`(runs + caches + outputs)
- `./scripts`  → `/scripts`  (entrypoint + wrappers)

所有 commands 都在 container 内运行，但 files 会通过这些 mounts 写入到你的 host。

---

## Prerequisites

- Docker + Docker Compose
- GPU + toolkit（用于 `gpus: all`）
- SDXL base model，形式为以下之一: (a) `./models/base/` 下的 local `.safetensors`/diffusers dir，或 (b) Hugging Face repo id（例如 `stabilityai/sdxl-turbo`）
 - `./datasets/<subject>/images/` 下的 small dataset
---

Highlights:
- **Reproducible**: 一切都在 container 内运行（不需要 local Python env）。
- **Simple**: 用一条 command 完成（可选的）image caption 生成 + train。
- **Safe defaults** for few-shot SDXL LoRA。
- **Includes inference**: 使用 `diffusers` 通过 LoRA 进行 SDXL txt2img。

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

如果你想在每张 image 旁生成同 basename 的 `.txt` captions:

```bash
# caption
docker compose run  \
--rm trainer caption  \  
--images /datasets/title \   
--prefix sksSubject    \
--overwrite
```

## Inference (SDXL txt2img with LoRA)

使用 trained LoRA generate images:

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

LoRA (Low-Rank Adaptation) 通过向选定的 weight matrices 添加 low-rank update，同时保持 base weights frozen，来 fine-tune diffusion model。

对于 weight matrix W，LoRA 学习:

ΔW = (α / r) * (B @ A)

其中:

r 是 rank (--network-dim)

α 是 scaling factor (--network-alpha)

A 和 B 是 low-rank trainable matrices

在 inference time，effective weight 变为:

W' = W + ΔW

此外，此 repo 允许你通过 --lora-scale 控制 LoRA 对 generation 的影响强度。

## License
- Apache License 2.0
