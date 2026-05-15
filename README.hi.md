---
layout: page
title: "🇮🇳 हिंदी"
permalink: /hi/
lang: hi
---

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

**SDXL LoRA adapters को train करने** और **SDXL inference चलाने** के लिए एक docker container।

यह repo “small image set” LoRA runs के लिए अनुकूलित है:
1) images को एक folder में डालें  
2) (वैकल्पिक रूप से) captions auto-generate करें  
3) LoRA को `./models/loras/` में train करें  
4) उसी LoRA से तुरंत images generate करें

---

## अंदर क्या है

- **GPU trainer container**
- **Command entrypoint**: `train` / `caption` / `infer`
- **LoRA training wrapper** 
- **Training launcher wrapper**
- **BLIP captioning tool**
- **Diffusers inference script**
- **CI के लिए CPU-only test container**

---

## Architecture / Mounts

`docker-compose.yml` local folders को container में mount करता है:

- `./models`   → `/models`   (base models + output LoRAs)
- `./datasets` → `/datasets` (आपकी raw images)
- `./workspace`→ `/workspace`(runs + caches + outputs)
- `./scripts`  → `/scripts`  (entrypoint + wrappers)

सभी commands container के अंदर चलते हैं, लेकिन files इन mounts के माध्यम से आपके host पर लिखी जाती हैं।

---

## Prerequisites

- Docker + Docker Compose
- GPU + toolkit (`gpus: all` के लिए)
- SDXL base model इनमें से किसी एक रूप में: (a) `./models/base/` के अंदर local `.safetensors`/diffusers dir, या (b) Hugging Face repo id (जैसे, `stabilityai/sdxl-turbo`)
 - `./datasets/<subject>/images/` के अंदर एक small dataset
---

Highlights:
- **Reproducible**: सब कुछ container के अंदर चलता है (local Python env की आवश्यकता नहीं)।
- **Simple**: images को (वैकल्पिक रूप से) caption करने + train करने के लिए एक command।
- **Safe defaults** few-shot SDXL LoRA के लिए।
- **Includes inference**: `diffusers` का उपयोग करके LoRA के साथ SDXL txt2img।

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

यदि आप हर image के पास उसी basename वाली `.txt` captions generate करना चाहते हैं:

```bash
# caption
docker compose run  \
--rm trainer caption  \  
--images /datasets/title \   
--prefix sksSubject    \
--overwrite
```

## Inference (SDXL txt2img with LoRA)

Trained LoRA के साथ images generate करें:

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

LoRA (Low-Rank Adaptation) base weights को frozen रखते हुए selected weight matrices में low-rank update जोड़कर diffusion model को fine-tune करता है।

किसी weight matrix W के लिए, LoRA यह सीखता है:

ΔW = (α / r) * (B @ A)

जहाँ:

r rank है (--network-dim)

α scaling factor है (--network-alpha)

A और B low-rank trainable matrices हैं

Inference time पर effective weight यह बन जाता है:

W' = W + ΔW

इसके अतिरिक्त, यह repo आपको --lora-scale के माध्यम से यह नियंत्रित करने देता है कि LoRA generation को कितनी मजबूती से प्रभावित करे।

## License
- Apache License 2.0
