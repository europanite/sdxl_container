#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
make_lora.sh - Train a LoRA from a small set of images.

Required:
  --base-model  /models/base/<model>.safetensors
  --images      /datasets/<subject>/images
  --run-name    e.g. my_subject

Optional (common):
  --concept-token   e.g. sksSubject
  --out-dir         default: /workspace/models/loras
  --caption-mode    none|blip (default: none)
  --sdxl            set SDXL-friendly defaults (1024px)
  --max-train-steps default: 1600
  --num-repeats     default: 20
  --network-dim     default: 16
  --network-alpha   default: 8
USAGE
}

BASE_MODEL=""
IMAGES_DIR=""
RUN_NAME=""
CONCEPT_TOKEN="sksSubject"
OUT_DIR="/workspace/models/loras"
CAPTION_MODE="none"
SDXL="0"
MAX_TRAIN_STEPS="1600"
NUM_REPEATS="20"
NETWORK_DIM="16"
NETWORK_ALPHA="8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model) BASE_MODEL="$2"; shift 2;;
    --images) IMAGES_DIR="$2"; shift 2;;
    --run-name) RUN_NAME="$2"; shift 2;;
    --concept-token) CONCEPT_TOKEN="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --caption-mode) CAPTION_MODE="$2"; shift 2;;
    --sdxl) SDXL="1"; shift 1;;
    --max-train-steps) MAX_TRAIN_STEPS="$2"; shift 2;;
    --num-repeats) NUM_REPEATS="$2"; shift 2;;
    --network-dim) NETWORK_DIM="$2"; shift 2;;
    --network-alpha) NETWORK_ALPHA="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$BASE_MODEL" || -z "$IMAGES_DIR" || -z "$RUN_NAME" ]]; then
  echo "Missing required arguments."
  usage
  exit 1
fi

if [[ ! -f "$BASE_MODEL" ]]; then
  echo "Base model not found: $BASE_MODEL"
  exit 1
fi

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "Images dir not found: $IMAGES_DIR"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/workspace/runs/${RUN_NAME}_${TS}"
DATA_DIR="${RUN_DIR}/dataset"
# sd-scripts "DreamBooth method" expects: <train_data_dir>/<repeats>_<token>/*.(png|jpg|webp) and optional captions
INSTANCE_DIR="${DATA_DIR}/${NUM_REPEATS}_${CONCEPT_TOKEN}"
mkdir -p "$RUN_DIR" "$DATA_DIR" "$INSTANCE_DIR" "$OUT_DIR" "/workspace/cache"

echo "Run dir: $RUN_DIR"
echo "Copying images into: $INSTANCE_DIR"
cp -a "${IMAGES_DIR}/." "$INSTANCE_DIR/"

# Sanity check: make sure we actually have images to train on
IMG_COUNT="$(find "$INSTANCE_DIR" -maxdepth 1 -type f \( \
  -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) | wc -l)"
if [[ "$IMG_COUNT" -eq 0 ]]; then
  echo "ERROR: No images found under: $INSTANCE_DIR" >&2
  echo "       Check --images path and volume mounts." >&2
  exit 2
fi
echo "Found $IMG_COUNT training images."

# Captioning (optional): writes one .txt per image (same basename)
if [[ "$CAPTION_MODE" == "blip" ]]; then
  echo "Captioning with BLIP..."
  python3 /scripts/caption_images.py \
    --images "$INSTANCE_DIR" \
    --prefix "${CONCEPT_TOKEN}" \
    --overwrite
else
  echo "Captioning disabled (mode: ${CAPTION_MODE})."
fi

# Training defaults
RESOLUTION="768"
if [[ "$SDXL" == "1" ]]; then
  RESOLUTION="1024"
fi

echo "Starting training..."
TRAIN_ARGS=(
  --base-model "$BASE_MODEL"
  --train-data-dir "${DATA_DIR}"
  --output-dir "$OUT_DIR"
  --output-name "${RUN_NAME}_${TS}"
  --resolution "$RESOLUTION"
  --max-train-steps "$MAX_TRAIN_STEPS"
  --num-repeats "$NUM_REPEATS"
  --network-dim "$NETWORK_DIM"
  --network-alpha "$NETWORK_ALPHA"
)
if [[ "$SDXL" == "1" ]]; then
  TRAIN_ARGS+=(--sdxl)
fi

bash /scripts/train_network.sh "${TRAIN_ARGS[@]}"

echo "Done."
echo "LoRA saved under: ${OUT_DIR}/${RUN_NAME}_${TS}*"
