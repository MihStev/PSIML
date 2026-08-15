#!/usr/bin/env python
"""Extract clips from a static demo page back out as mp4 tiles + poster PNGs, so the
same demo can live inside the .pptx deck as click-to-play shapes (PowerPoint has no
HTML embedding; a d-pad of movie tiles is the closest native equivalent).

Reads the JSON blob baked into the page (`const DATA=...`), decodes the base64
frames, and writes per-action mp4s plus button-styled posters (arrow glyph over the
darkened start frame -- drawn with PIL polygons, no font dependency).

Base64 format differs between demo builds (bit us once):
  - demo1 (`logs/demo/index.html`, 64x64 PNG): plain base64, no prefix
  - demo2 (`logs/demo2/index.html`, 256x256 JPEG): 1-char mime marker prefix ('p'/'j')
This script handles demo1 (the deck uses it: 4 scenes, native pixels, no ESRGAN).

Usage:
    python extract_demo1_widget.py --html /home/mls10/logs/demo/index.html \
        --scene 204 --out /home/mls10/presentation/widget
"""
import argparse
import base64
import io
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ACCENT = (56, 214, 196)   # deck accent teal
DARK = (10, 12, 15)       # deck ground


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--html", default="/home/mls10/logs/demo/index.html")
    p.add_argument("--scene", default="204",
                   help="scene id key in the page's DATA blob")
    p.add_argument("--out", default="/home/mls10/presentation/widget")
    p.add_argument("--scale", type=int, default=8,
                   help="nearest-neighbor upscale for tiles/posters (honest, no detail added)")
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def load_data(path):
    html = open(path, encoding="utf-8").read()
    i = html.find("DATA=")
    assert i != -1, "no DATA= in page"
    data, _ = json.JSONDecoder().raw_decode(html[i + 5:])
    return data


def frames_of(lst, scale):
    out = []
    for s in lst:
        im = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
        w, h = im.size
        im = im.resize((w * scale, h * scale), Image.NEAREST)
        out.append(np.asarray(im))
    return out


def arrow_poly(cx, cy, s, direction):
    if direction == "up":
        return [(cx, cy - s), (cx + s * 0.8, cy + s * 0.6), (cx, cy + s * 0.15),
                (cx - s * 0.8, cy + s * 0.6)]
    if direction == "down":
        return [(cx, cy + s), (cx + s * 0.8, cy - s * 0.6), (cx, cy - s * 0.15),
                (cx - s * 0.8, cy - s * 0.6)]
    if direction == "left":
        return [(cx - s, cy), (cx + s * 0.6, cy - s * 0.8), (cx + s * 0.15, cy),
                (cx + s * 0.6, cy + s * 0.8)]
    if direction == "right":
        return [(cx + s, cy), (cx - s * 0.6, cy - s * 0.8), (cx - s * 0.15, cy),
                (cx - s * 0.6, cy + s * 0.8)]
    raise ValueError(direction)


def make_poster(bg, kind, path):
    im = Image.fromarray(bg).convert("RGB")
    W = im.size[0]
    im = Image.blend(im, Image.new("RGB", im.size, DARK), 0.45)
    d = ImageDraw.Draw(im, "RGBA")
    cx = cy = W // 2
    r = int(W * 0.23)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=DARK + (175,))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ACCENT + (210,), width=4)
    if kind in ("up", "down", "left", "right"):
        d.polygon(arrow_poly(cx, cy, int(W * 0.12), kind), fill=ACCENT + (255,))
    elif kind == "still":
        s = int(W * 0.085)
        d.rectangle([cx - s, cy - s, cx + s, cy + s], outline=ACCENT + (255,), width=10)
    elif kind == "chain":
        step = int(W * 0.11)
        x0 = cx - 1.5 * step
        for j, k in enumerate(["up", "down", "left", "right"]):
            d.polygon(arrow_poly(x0 + j * step, cy, int(W * 0.05), k),
                      fill=ACCENT + (255,))
    im.save(path)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load_data(args.html)
    assert args.scene in data, f"scene {args.scene} not in {list(data)}"
    scene = data[args.scene]

    ctx = frames_of(scene["context"], args.scale)
    last = ctx[-1]
    Image.fromarray(last).save(os.path.join(args.out, "screen_context.png"))

    for name in ["up", "down", "left", "right", "still"]:
        gen = frames_of(scene["anchored"][name], args.scale)
        imageio.mimsave(os.path.join(args.out, f"tile_{name}.mp4"), gen,
                        fps=args.fps, macro_block_size=1)
        make_poster(last, name, os.path.join(args.out, f"tile_{name}.png"))
        print(name, len(gen), "frames")

    chain = list(ctx)
    for b in scene["chain"]:
        chain += frames_of(b["frames"], args.scale)
    imageio.mimsave(os.path.join(args.out, "tile_chain.mp4"), chain,
                    fps=args.fps, macro_block_size=1)
    make_poster(last, "chain", os.path.join(args.out, "tile_chain.png"))
    print("chain", len(chain), "frames")


if __name__ == "__main__":
    main()
