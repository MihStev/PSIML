#!/usr/bin/env python
"""Generative 4x super-resolution of one clip, via Stable Diffusion x4 upscaler.

Why this and not Real-ESRGAN: ESRGAN is a regression model, so it resolves ambiguity by
averaging and the result goes waxy -- plasticine surfaces, no grain. A diffusion upscaler
SAMPLES the missing detail instead of averaging it, so it puts back texture: cloth weave,
plastic sheen, the speckle of a table surface.

The flip side is that it invents MORE, not less. Nothing here is our model's output and
none of it enters any reported number -- every metric in the project (PSNR 18.56 against
the 22.74 dB VAE ceiling) refers to the raw 64x64 frames. Label it wherever it is shown.

Temporal note: frames are upscaled independently, so sampled texture differs slightly
between frames and the clip can shimmer. Fixing that needs a video SR model. Using the
SAME seed for every frame (default here) keeps the noise draw fixed and reduces it a lot.

Usage:
    python diffusion_sr_video.py --src clip.mp4
    python diffusion_sr_video.py --src clip.mp4 --compare   # ESRGAN left, diffusion right
"""
import argparse
import os

import imageio
import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--prompt", default="a robot arm and small objects on a table, "
                                        "sharp photo, fine texture, natural detail")
    p.add_argument("--negative", default="blurry, waxy, plastic, smooth, cartoon, painting")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--noise_level", type=int, default=20,
                    help="how much the upscaler assumes the input is degraded; higher lets it "
                         "invent more (and drift further from our output)")
    p.add_argument("--seed", type=int, default=0,
                    help="SAME seed for every frame -> much less shimmer between frames")
    p.add_argument("--max_frames", type=int, default=0, help="0 = all")
    p.add_argument("--compare", default=None,
                    help="path to an ESRGAN version of the same clip; puts it on the left")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")
    from diffusers import StableDiffusionUpscalePipeline

    print("  loading stabilityai/stable-diffusion-x4-upscaler ...", flush=True)
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    frames = [np.asarray(f)[..., :3] for f in imageio.mimread(args.src, memtest=False)]
    if args.max_frames:
        frames = frames[:args.max_frames]
    h, w = frames[0].shape[:2]
    print(f"  {os.path.basename(args.src)}: {len(frames)} frejmova {w}x{h} -> {w*4}x{h*4}", flush=True)

    cmp_frames = None
    if args.compare:
        cmp_frames = [np.asarray(f)[..., :3] for f in imageio.mimread(args.compare, memtest=False)]

    out_frames = []
    for i, f in enumerate(frames):
        g = torch.Generator(device=device).manual_seed(args.seed)   # fixed -> stable texture
        sr = pipe(prompt=args.prompt, negative_prompt=args.negative,
                  image=Image.fromarray(f), num_inference_steps=args.steps,
                  guidance_scale=args.guidance, noise_level=args.noise_level,
                  generator=g).images[0]
        sr = np.asarray(sr)
        if cmp_frames is not None:
            left = np.asarray(Image.fromarray(cmp_frames[i]).resize(sr.shape[1::-1], Image.LANCZOS))
            sr = np.concatenate([left, sr], axis=1)
        out_frames.append(sr)
        if (i + 1) % 5 == 0 or i == len(frames) - 1:
            print(f"    {i+1}/{len(frames)}", flush=True)

    out = args.out or os.path.splitext(args.src)[0] + "_diffsr.mp4"
    imageio.mimsave(out, out_frames, fps=args.fps, macro_block_size=1)
    print(f"  -> {out}   ({out_frames[0].shape[1]}x{out_frames[0].shape[0]})", flush=True)
    print("  NAPOMENA: difuzioni upscaler IZMISLJA teksturu. Nije nas model, ne ulazi ni u jedan broj.",
          flush=True)


if __name__ == "__main__":
    main()
