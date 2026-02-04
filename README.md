# Few-shot LoRA Training System (Docker Compose)

This repository is a **reproducible, Docker-first** template for training a LoRA from a small set of images ("few-shot") on top of a general-purpose base model (e.g., SDXL), plus a small inference script to generate images with the trained LoRA.

Highlights:
- **Reproducible**: everything runs inside a container (no local Python env needed).
- **Simple**: one command to (optionally) caption images + train.
- **Safe defaults** for few-shot SDXL LoRA.
- **Includes inference**: SDXL txt2img with LoRA using `diffusers`.

---

## Requirements

- Docker + Docker Compose v2
- A GPU-enabled Docker runtime is strongly recommended for training.
  - The default `docker-compose.yml` sets `gpus: all`.
  - If you do not have GPU support, remove the `gpus: all` line (training may still be impractically slow).

---

## Directory layout

Place your files like this:

```
.
├── docker-compose.yml
├── docker-compose.test.yml
├── service/
│   ├── Dockerfile
│   └── Dockerfile.test
├── scripts/
│   ├── entrypoint.sh
│   ├── make_lora.sh
│   ├── train_network.sh
│   ├── infer_sdxl_lora.py
│   └── caption_images.py
└── work/
    ├── models/
    │   ├── base/
    │   │   └── sd_xl_base_1.0.safetensors
    │   └── loras/
    ├── datasets/
    │   └── my_subject/
    │       └── images/
    │           ├── 0001.webp
    │           └── ...
    ├── runs/
    ├── outputs/
    └── cache/
```

Notes:
- Base model goes under `work/models/base/`
- Training images go under `work/datasets/<subject>/images/`

---

## Build

```
docker compose build --no-cache trainer
```

Tip: `--no-cache` is recommended after changing dependencies.

---

## Train (caption + LoRA)

Example for SDXL:

```
docker compose run --rm trainer train   --base-model /work/models/base/sd_xl_base_1.0.safetensors   --images /work/datasets/my_subject/images   --run-name my_subject   --sdxl   --caption-mode blip   --concept-token sksSubject   --max-train-steps 1600   --num-repeats 20   --network-dim 16   --network-alpha 8
```

Outputs:
- LoRA weights: `work/models/loras/<run_name>_<timestamp>*.safetensors`
- Run artifacts (dataset + logs): `work/runs/<run_name>_<timestamp>/...`

---

## Caption only (BLIP)

If you want to generate `.txt` captions next to each image (same basename):

```
docker compose run --rm trainer caption   --images /work/datasets/my_subject/images   --prefix sksSubject   --overwrite
```

---

## Inference (SDXL txt2img with LoRA)

Generate images with the trained LoRA:

```
docker compose run --rm trainer infer   --base-model /work/models/base/sd_xl_base_1.0.safetensors   --lora /work/models/loras/my_subject_20260203_123456.safetensors   --prompt "sksSubject portrait photo, ultra detailed, 85mm"   --negative-prompt ""   --out-dir /work/outputs   --num-images 4   --steps 30   --cfg 7.0   --width 1024   --height 1024   --lora-scale 0.8   --seed 42
```

Helpful flags (especially when memory is tight):
- `--cpu-offload`
- `--attention-slicing`
- `--device "cuda:0"` (or `cpu`)

### About `--base-model`

`--base-model` can be:
- A **diffusers** directory (recommended), or
- A **single `.safetensors`** file (only if your installed diffusers supports `from_single_file` for the SDXL pipeline)

If your environment does not support single-file loading, convert your checkpoint to a diffusers directory and pass the directory.

---

## How it works

The container entrypoint routes subcommands:

- `train` → `scripts/make_lora.sh` (prepares dataset, optional captioning, then trains)
- `caption` → `scripts/caption_images.py`
- `infer` → `scripts/infer_sdxl_lora.py`
- `bash` / `sh` → interactive shell

Examples:

```
docker compose run --rm trainer bash
docker compose run --rm trainer python3 -V
```

---

## Dataset format (DreamBooth method)

Training uses the folder format expected by sd-scripts (“DreamBooth method”):

```
/work/runs/<run_name>_<timestamp>/dataset/<repeats>_<concept-token>/
```

`make_lora.sh` creates this structure automatically by copying your `--images` directory into it.

---

## Training defaults (what the scripts set)

The training wrapper uses a conservative baseline suitable for “few images” LoRA:
- `bf16` mixed precision (via `accelerate`)
- batch size 1
- cosine LR schedule
- caching latents + bucketed resolution
- SDXL mode switches to `sdxl_train_network.py` automatically

It also sets `PYTORCH_ALLOC_CONF=expandable_segments:True` by default to reduce CUDA allocator fragmentation.

Additionally, the wrapper will add memory-saving flags only when supported by your installed sd-scripts version (it checks `--help` output), e.g.:
- `--gradient_checkpointing`
- `--xformers`
- `--sdpa`
- `--network_train_unet_only`

---

## Cache locations

The compose file mounts `./work` and sets:
- `HF_HOME=/work/cache/hf`
- `PIP_CACHE_DIR=/work/cache/pip`

This keeps downloads persistent across container runs.

---

## Customizing upstream repos / versions

`docker-compose.yml` passes build args so you can pin forks/refs:

- `SD_SCRIPTS_REPO` / `SD_SCRIPTS_REF`
- `KOHYA_REPO` / `KOHYA_REF`
- `TORCH_INDEX_URL`

Example (pin to a specific sd-scripts tag/branch):

```
SD_SCRIPTS_REF=v0.8.7 docker compose build --no-cache trainer
```

---

## Tests (service)

A lightweight test container is provided:

```
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit
```

Or run once:

```
docker compose -f docker-compose.test.yml run --rm service_test
```

---

## Security

See `SECURITY.md`.

---

## License / attribution

This repo is a **template**. The training toolchain is pulled at build time from upstream repositories (sd-scripts and kohya_ss). Please comply with the licenses of all upstream dependencies and the base model you use.

# Few-shot LoRA Training System (Docker Compose)

This template trains a LoRA from a small set of images ("few-shot") using a general-purpose base model (e.g., SDXL). It is designed to be:

- **Reproducible**: everything runs in Docker.
- **Simple**: one command to caption (optional) + train.
- **Safe defaults** for SDXL few-shot LoRA.

## Directory layout

Place your files like this:


```bash
# train
docker compose run --rm trainer train \
--base-model /models/base/sd_xl_base_1.0.safetensors \
--images /datasets/chun/images \
--run-name title \
--sdxl \
--caption-mode blip \
--concept-token sksSubject \
--max-train-steps 1600 \
--num-repeats 20 \
--network-dim 16 \
--network-alpha 8
```