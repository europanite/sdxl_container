#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-bash}"
shift || true

case "$cmd" in
  train)
    exec bash /scripts/make_lora.sh "$@"
    ;;
  infer)
    exec python3 /scripts/infer_sdxl_lora.py "$@"
    ;;
  caption)
    exec python3 /scripts/caption_images.py "$@"
    ;;
  bash|sh)
    exec bash
    ;;
  *)
    # Allow arbitrary commands (e.g., "python3 -V")
    exec "$cmd" "$@"
    ;;
esac
