#!/usr/bin/env python
"""Honest upscaling of an output clip, for projection.

Deliberately NOT a learned super-resolution model. A learned SR would invent
high-frequency detail the model never produced, and this project's central
fidelity claim is that we sit at 81% of a measured 22.74 dB VAE ceiling --
showing a sharpened frame and calling it our output would misrepresent that.

Lanczos resampling adds no information; every pixel still traces back to the
model. The clip just becomes legible on a projector.

Usage:
    python upscale_video.py --src <in.mp4> --scale 8
    python upscale_video.py --src <in.mp4> --scale 8 --fps 8 --side_by_side
"""
import argparse
import os

import imageio
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", default=None, help="default: <src stem>_x<scale>.mp4")
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--side_by_side", action="store_true",
                    help="left: nearest (shows the true pixel grid), right: Lanczos (smooth)")
    return p.parse_args()


def main():
    args = parse_args()
    frames = imageio.mimread(args.src, memtest=False)
    frames = [np.asarray(f)[..., :3] for f in frames]
    h, w = frames[0].shape[:2]
    W, H = w * args.scale, h * args.scale
    print(f"  {os.path.basename(args.src)}: {len(frames)} frejmova {w}x{h} -> {W}x{H}", flush=True)

    out_frames = []
    for f in frames:
        im = Image.fromarray(f)
        lanc = im.resize((W, H), Image.LANCZOS)
        if args.side_by_side:
            near = im.resize((W, H), Image.NEAREST)
            canvas = Image.new("RGB", (W * 2, H))
            canvas.paste(near, (0, 0))
            canvas.paste(lanc, (W, 0))
            out_frames.append(np.asarray(canvas))
        else:
            out_frames.append(np.asarray(lanc))

    out = args.out or os.path.splitext(args.src)[0] + f"_x{args.scale}.mp4"
    imageio.mimsave(out, out_frames, fps=args.fps, macro_block_size=1)
    print("  ->", out, flush=True)


if __name__ == "__main__":
    main()
