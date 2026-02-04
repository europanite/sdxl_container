#!/usr/bin/env bash
set -euo pipefail

# Wrapper around sd-scripts train_network.py (installed at /opt/sd-scripts)
# Adjust arguments as needed. This is a sane baseline for "few images" LoRA.

BASE_MODEL=""
TRAIN_DATA_DIR=""
OUTPUT_DIR=""
OUTPUT_NAME=""
RESOLUTION="768"
SDXL="0"
MAX_TRAIN_STEPS="1600"
NUM_REPEATS="20"
NETWORK_DIM="16"
NETWORK_ALPHA="8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) BASE_MODEL="$2"; shift 2;;
    --train-data-dir) TRAIN_DATA_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --output-name) OUTPUT_NAME="$2"; shift 2;;
    --resolution) RESOLUTION="$2"; shift 2;;
    --max-train-steps) MAX_TRAIN_STEPS="$2"; shift 2;;
    --num-repeats) NUM_REPEATS="$2"; shift 2;;
    --network-dim) NETWORK_DIM="$2"; shift 2;;
    --network-alpha) NETWORK_ALPHA="$2"; shift 2;;
    --sdxl) SDXL="1"; shift 1;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$BASE_MODEL" || -z "$TRAIN_DATA_DIR" || -z "$OUTPUT_DIR" || -z "$OUTPUT_NAME" ]]; then
  echo "train_network.sh: missing required args"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Reduce CUDA allocator fragmentation (recommended by PyTorch when OOM happens with small allocations)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Select the correct training entrypoint.
# sd-scripts trains SDXL LoRA with sdxl_train_network.py (not train_network.py + --sdxl).
TRAIN_SCRIPT="/opt/sd-scripts/train_network.py"
CLIP_SKIP_ARG="--clip_skip=1"
if [[ "$SDXL" == "1" ]]; then
  TRAIN_SCRIPT="/opt/sd-scripts/sdxl_train_network.py"
  CLIP_SKIP_ARG=""
  if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: SDXL mode was requested (--sdxl), but $TRAIN_SCRIPT was not found." >&2
    exit 2
  fi
fi

# Add memory-saving flags only when supported by the installed sd-scripts
HELP_TEXT="$(python3 "$TRAIN_SCRIPT" --help 2>&1 || true)"
EXTRA_ARGS=()
grep -q -- '--network_train_unet_only' <<<"$HELP_TEXT" && EXTRA_ARGS+=(--network_train_unet_only)
grep -q -- '--gradient_checkpointing' <<<"$HELP_TEXT" && EXTRA_ARGS+=(--gradient_checkpointing)
grep -q -- '--sdpa' <<<"$HELP_TEXT" && EXTRA_ARGS+=(--sdpa)
grep -q -- '--xformers' <<<"$HELP_TEXT" && EXTRA_ARGS+=(--xformers)

# NOTE: sd-scripts expects captions as .txt next to images if you use captioning.
# For DreamBooth method, repeats are encoded in the dataset subfolder name: <repeats>_<token>.
# (make_lora.sh creates that folder automatically.)
accelerate launch --num_processes 1 --mixed_precision bf16 "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path="$BASE_MODEL" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --output_name="$OUTPUT_NAME" \
  --resolution="$RESOLUTION" \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler=cosine \
  --lr_warmup_steps=0 \
  --network_module=networks.lora \
  --network_dim="$NETWORK_DIM" \
  --network_alpha="$NETWORK_ALPHA" \
  --mixed_precision=bf16 \
  --save_precision=bf16 \
  --save_model_as=safetensors \
  --prior_loss_weight=1.0 \
  $CLIP_SKIP_ARG \
  --min_snr_gamma=5 \
  --cache_latents \
  --cache_latents_to_disk \
  --enable_bucket \
  --bucket_no_upscale \
  "${EXTRA_ARGS[@]}" \
  --caption_extension=.txt
