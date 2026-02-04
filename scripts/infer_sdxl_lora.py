#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SDXL txt2img inference with LoRA (diffusers).")
    p.add_argument("--base-model", required=True, help="Base SDXL: dir or .safetensors")
    p.add_argument("--lora", required=True, help="LoRA weights: .safetensors")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--out-dir", default="/workspace/outputs")
    p.add_argument("--num-images", type=int, default=1)
    p.add_argument("--seed", type=int, default=-1, help="-1=random")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=7.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--lora-scale", type=float, default=0.8)
    p.add_argument("--device", default="", help='e.g. "cuda", "cuda:0", "cpu" (default: auto)')
    p.add_argument("--cpu-offload", action="store_true")
    p.add_argument("--attention-slicing", action="store_true")
    return p


def load_pipe(base_model: str, device: str, dtype: torch.dtype):
    from diffusers import StableDiffusionXLPipeline  # type: ignore

    bm = Path(base_model)
    if bm.is_dir():
        pipe = StableDiffusionXLPipeline.from_pretrained(str(bm), torch_dtype=dtype, use_safetensors=True)
    else:
        if not hasattr(StableDiffusionXLPipeline, "from_single_file"):
            raise SystemExit("diffusers does not support from_single_file. Pass a diffusers-format directory to --base-model.")
        pipe = StableDiffusionXLPipeline.from_single_file(str(bm), torch_dtype=dtype)

    if device.startswith("cuda"):
        pipe.to(device)
    else:
        pipe.to("cpu")
    return pipe


def load_lora(pipe, lora_path: str):
    lp = Path(lora_path)
    # For compatibility, load using the "parent directory + weight_name" style.
    pipe.load_lora_weights(str(lp.parent), weight_name=lp.name)


def main() -> None:
    args = build_argparser().parse_args()

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipe(args.base_model, device, dtype)
    load_lora(pipe, args.lora)

    if args.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    if args.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()

    ts = time.strftime("%Y%m%d_%H%M%S")

    for i in range(args.num_images):
        seed = args.seed
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        else:
            seed = seed + i

        gen_device = device if device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Apply LoRA scale via cross_attention_kwargs (to absorb diffusers version differences).
        extra = {"cross_attention_kwargs": {"scale": float(args.lora_scale)}}

        images = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or None,
            width=int(args.width),
            height=int(args.height),
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.cfg),
            generator=generator,
            **extra,
        ).images

        out_path = out_dir / f"sdxl_lora_{ts}_{i:02d}_seed{seed}.png"
        images[0].save(out_path)
        print(out_path)


if __name__ == "__main__":
    main()
