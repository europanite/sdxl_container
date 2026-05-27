---
layout: page
title: "🇪🇸 Español"
permalink: /es/
lang: es
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

Un docker container para **train SDXL LoRA adapters** y **run SDXL inference**.

Este repo está optimizado para LoRA runs con “small image set”:
1) coloca images en un folder  
2) (opcionalmente) auto-generate captions  
3) train un LoRA en `./models/loras/`  
4) generate images inmediatamente con ese LoRA

---

## What’s inside

- **GPU trainer container**
- **Command entrypoint**: `train` / `caption` / `infer`
- **LoRA training wrapper** 
- **Training launcher wrapper**
- **BLIP captioning tool**
- **Diffusers inference script**
- **CPU-only test container** para CI

---

## Architecture / Mounts

`docker-compose.yml` monta local folders dentro del container:

- `./models`   → `/models`   (base models + output LoRAs)
- `./datasets` → `/datasets` (your raw images)
- `./workspace`→ `/workspace`(runs + caches + outputs)
- `./scripts`  → `/scripts`  (entrypoint + wrappers)

Todos los commands se ejecutan dentro del container, pero los files se escriben en tu host mediante estos mounts.

---

## Prerequisites

- Docker + Docker Compose
- GPU + toolkit (para `gpus: all`)
- Un SDXL base model como una de estas opciones: (a) local `.safetensors`/diffusers dir bajo `./models/base/`, o (b) un Hugging Face repo id (por ejemplo, `stabilityai/sdxl-turbo`)
 - Un small dataset bajo `./datasets/<subject>/images/`
---

Highlights:
- **Reproducible**: todo se ejecuta dentro de un container (no se necesita local Python env).
- **Simple**: un command para caption images (opcionalmente) + train.
- **Safe defaults** para few-shot SDXL LoRA.
- **Includes inference**: SDXL txt2img con LoRA usando `diffusers`.

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

Si quieres generate `.txt` captions junto a cada image (con el mismo basename):

```bash
# caption
docker compose run  \
--rm trainer caption  \  
--images /datasets/title \   
--prefix sksSubject    \
--overwrite
```

## Inference (SDXL txt2img with LoRA)

Generate images con el trained LoRA:

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

LoRA (Low-Rank Adaptation) fine-tunes un diffusion model agregando un low-rank update a weight matrices seleccionadas mientras mantiene los base weights frozen.

Para una weight matrix W, LoRA aprende:

ΔW = (α / r) * (B @ A)

Donde:

r es el rank (--network-dim)

α es el scaling factor (--network-alpha)

A y B son las low-rank trainable matrices

En inference time, el effective weight se convierte en:

W' = W + ΔW

Además, este repo te permite controlar qué tan fuerte influye LoRA en la generation mediante --lora-scale.

## License
- Apache License 2.0
