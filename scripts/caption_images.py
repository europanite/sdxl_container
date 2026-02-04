import argparse
import os
from pathlib import Path

from PIL import Image


def try_load_blip():
    """
    Lazy import so the container can still run training even if caption deps are missing.
    """
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore

        return BlipProcessor, BlipForConditionalGeneration
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "BLIP captioning dependencies are missing or failed to import. "
            "Install transformers + sentencepiece, and ensure torch is available.\n"
            f"Original error: {e}"
        ) from e


def iter_images(images_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Directory containing training images")
    ap.add_argument("--prefix", default="", help="Prefix token (e.g., 'sksSubject') added to captions")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt captions")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        raise SystemExit(f"Not a directory: {images_dir}")

    BlipProcessor, BlipForConditionalGeneration = try_load_blip()

    # Small caption model (base) to keep this lightweight
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)

    prefix = (args.prefix.strip() + " ") if args.prefix.strip() else ""

    for img_path in iter_images(images_dir):
        cap_path = img_path.with_suffix(".txt")
        if cap_path.exists() and not args.overwrite:
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()

        # Very simple formatting: "<prefix><caption>"
        text = f"{prefix}{caption}\n"
        cap_path.write_text(text, encoding="utf-8")
        print(f"[captioned] {img_path.name} -> {cap_path.name}")

    print("Captioning finished.")


if __name__ == "__main__":
    main()
