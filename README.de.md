[English](README.md) | [हिन्दी](README.hi.md) | [日本語](README.ja.md) | [简体中文](README.zh-CN.md) | [Español](README.es.md) | [Português (Brasil)](README.pt-BR.md) | [한국어](README.ko.md) | [Deutsch](README.de.md) | [Français](README.fr.md)

> **Hinweis:** Dies ist eine übersetzte Version von `README.md`. Das englische README ist die maßgebliche Quelle.

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

Ein docker container, um **SDXL LoRA adapters zu trainieren** und **SDXL inference auszuführen**.

Dieses repo ist für LoRA runs mit “small image set” optimiert:
1) images in einen folder legen  
2) (optional) captions auto-generate  
3) ein LoRA nach `./models/loras/` trainieren  
4) sofort images mit diesem LoRA generate

---

## What’s inside

- **GPU trainer container**
- **Command entrypoint**: `train` / `caption` / `infer`
- **LoRA training wrapper** 
- **Training launcher wrapper**
- **BLIP captioning tool**
- **Diffusers inference script**
- **CPU-only test container** für CI

---

## Architecture / Mounts

`docker-compose.yml` mountet local folders in den container:

- `./models`   → `/models`   (base models + output LoRAs)
- `./datasets` → `/datasets` (your raw images)
- `./workspace`→ `/workspace`(runs + caches + outputs)
- `./scripts`  → `/scripts`  (entrypoint + wrappers)

Alle commands laufen im container, aber files werden über diese mounts auf deinen host geschrieben.

---

## Prerequisites

- Docker + Docker Compose
- GPU + toolkit (für `gpus: all`)
- Ein SDXL base model als entweder: (a) local `.safetensors`/diffusers dir unter `./models/base/`, oder (b) eine Hugging Face repo id (z. B. `stabilityai/sdxl-turbo`)
 - Ein small dataset unter `./datasets/<subject>/images/`
---

Highlights:
- **Reproducible**: alles läuft in einem container (keine local Python env erforderlich).
- **Simple**: ein command, um images (optional) zu caption + zu trainieren.
- **Safe defaults** für few-shot SDXL LoRA.
- **Includes inference**: SDXL txt2img mit LoRA über `diffusers`.

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

Wenn du `.txt` captions neben jeder image mit demselben basename generate möchtest:

```bash
# caption
docker compose run  \
--rm trainer caption  \  
--images /datasets/title \   
--prefix sksSubject    \
--overwrite
```

## Inference (SDXL txt2img with LoRA)

Generate images mit dem trained LoRA:

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

LoRA (Low-Rank Adaptation) fine-tunes ein diffusion model, indem es einen low-rank update zu ausgewählten weight matrices hinzufügt, während die base weights frozen bleiben.

Für eine weight matrix W lernt LoRA:

ΔW = (α / r) * (B @ A)

Dabei gilt:

r ist der rank (--network-dim)

α ist der scaling factor (--network-alpha)

A und B sind die low-rank trainable matrices

Zur inference time wird das effective weight zu:

W' = W + ΔW

Zusätzlich kannst du in diesem repo über --lora-scale steuern, wie stark das LoRA die generation beeinflusst.

## License
- Apache License 2.0
