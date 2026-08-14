#!/usr/bin/env python
"""Learned 4x super-resolution of one output clip, via Real-ESRGAN (spandrel loader).

This is NOT our model and NOT part of any reported number. Real-ESRGAN was trained
on natural images and it HALLUCINATES detail: edges it invents were never predicted
by the world model. It is here to make one clip look strong on a projector and to
show what a proper SR stage could add on top -- it must be labelled as such wherever
it is shown.

Every metric in the project (PSNR 18.56, the 22.74 dB VAE ceiling) refers to the raw
64x64 output, never to this.

Usage:
    python real_sr_video.py --src clip.mp4 --weights /tmp/resrgan.pth
    python real_sr_video.py --src clip.mp4 --compare      # Lanczos | Real-ESRGAN
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
    p.add_argument("--weights", default="/tmp/resrgan.pth")
    p.add_argument("--out", default=None)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--tile", type=int, default=0, help="0 = whole frame at once (64x64 is tiny)")
    p.add_argument("--compare", action="store_true",
                    help="side by side: honest Lanczos left, hallucinated Real-ESRGAN right")
    p.add_argument("--label", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from spandrel import ImageModelDescriptor, ModelLoader
    model = ModelLoader().load_from_file(args.weights)
    assert isinstance(model, ImageModelDescriptor), "not an image model"
    model.to(device).eval()
    scale = model.scale
    print(f"  Real-ESRGAN loaded: x{scale}, {model.architecture.name if hasattr(model,'architecture') else ''}",
          flush=True)

    frames = [np.asarray(f)[..., :3] for f in imageio.mimread(args.src, memtest=False)]
    h, w = frames[0].shape[:2]
    print(f"  {os.path.basename(args.src)}: {len(frames)} frejmova {w}x{h}", flush=True)

    out_frames = []
    with torch.no_grad():
        for f in frames:
            t = torch.from_numpy(f).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
            sr = model(t).clamp(0, 1)
            sr = (sr[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            H, W = sr.shape[:2]
            if args.compare:
                lz = np.asarray(Image.fromarray(f).resize((W, H), Image.LANCZOS))
                canvas = np.concatenate([lz, sr], axis=1)
                out_frames.append(canvas)
            else:
                out_frames.append(sr)

    out = args.out or os.path.splitext(args.src)[0] + f"_realsr_x{scale}.mp4"
    imageio.mimsave(out, out_frames, fps=args.fps, macro_block_size=1)
    print(f"  -> {out}   ({out_frames[0].shape[1]}x{out_frames[0].shape[0]})", flush=True)
    print("  NAPOMENA: Real-ESRGAN izmislja detalje. Nije nas model i ne ulazi ni u jedan broj.",
          flush=True)


if __name__ == "__main__":
    main()
