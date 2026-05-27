---
layout: page
title: "🇫🇷 Français"
permalink: /fr/
lang: fr
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

Un docker container pour **train SDXL LoRA adapters** et **run SDXL inference**.

Ce repo est optimisé pour les LoRA runs avec un “small image set”:
1) placez les images dans un folder  
2) (facultatif) auto-generate captions  
3) train un LoRA dans `./models/loras/`  
4) generate images immédiatement avec ce LoRA

---

## What’s inside

- **GPU trainer container**
- **Command entrypoint**: `train` / `caption` / `infer`
- **LoRA training wrapper** 
- **Training launcher wrapper**
- **BLIP captioning tool**
- **Diffusers inference script**
- **CPU-only test container** pour la CI

---

## Architecture / Mounts

`docker-compose.yml` monte les local folders dans le container:

- `./models`   → `/models`   (base models + output LoRAs)
- `./datasets` → `/datasets` (your raw images)
- `./workspace`→ `/workspace`(runs + caches + outputs)
- `./scripts`  → `/scripts`  (entrypoint + wrappers)

Toutes les commands s’exécutent dans le container, mais les files sont écrits sur votre host via ces mounts.

---

## Prerequisites

- Docker + Docker Compose
- GPU + toolkit (pour `gpus: all`)
- Un SDXL base model sous l’une des formes suivantes: (a) local `.safetensors`/diffusers dir sous `./models/base/`, ou (b) un Hugging Face repo id (par exemple, `stabilityai/sdxl-turbo`)
 - Un small dataset sous `./datasets/<subject>/images/`
---

Highlights:
- **Reproducible**: tout s’exécute dans un container (aucun local Python env nécessaire).
- **Simple**: une seule command pour caption images (facultatif) + train.
- **Safe defaults** pour few-shot SDXL LoRA.
- **Includes inference**: SDXL txt2img avec LoRA via `diffusers`.

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

Si vous voulez generate des captions `.txt` à côté de chaque image (même basename):

```bash
# caption
docker compose run  \
--rm trainer caption  \  
--images /datasets/title \   
--prefix sksSubject    \
--overwrite
```

## Inference (SDXL txt2img with LoRA)

Generate images avec le trained LoRA:

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

LoRA (Low-Rank Adaptation) fine-tunes un diffusion model en ajoutant un low-rank update à des weight matrices sélectionnées tout en gardant les base weights frozen.

Pour une weight matrix W, LoRA apprend:

ΔW = (α / r) * (B @ A)

Où:

r est le rank (--network-dim)

α est le scaling factor (--network-alpha)

A et B sont les low-rank trainable matrices

Au moment de l’inference time, l’effective weight devient:

W' = W + ΔW

De plus, ce repo vous permet de contrôler l’intensité de l’influence du LoRA sur la generation via --lora-scale.

## License
- Apache License 2.0
