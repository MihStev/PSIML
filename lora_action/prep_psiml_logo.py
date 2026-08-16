#!/usr/bin/env python
"""Cut the PSIML logo's flat navy background out to transparency, with a
soft-edge falloff (not a hard chroma-key cutout), so it drops onto any deck
background -- including our title-slide gradient -- without a visible box.
"""
import numpy as np
from PIL import Image

SRC = "/home/mls10/presentation/Screenshot 2026-08-15 194022.png"
OUT = "/home/mls10/presentation/psiml_logo.png"

BG = np.array([12, 34, 51], dtype=np.float32)
TOL_IN, TOL_OUT = 8.0, 40.0   # fully transparent below TOL_IN, fully opaque above TOL_OUT

im = Image.open(SRC).convert("RGB")
arr = np.asarray(im).astype(np.float32)
dist = np.linalg.norm(arr - BG, axis=-1)
alpha = np.clip((dist - TOL_IN) / (TOL_OUT - TOL_IN), 0.0, 1.0)
alpha = (alpha * 255).astype(np.uint8)

rgba = np.dstack([np.asarray(im), alpha])
Image.fromarray(rgba, mode="RGBA").save(OUT)
print("saved", OUT, "transparent bg pixels:", int((alpha == 0).sum()), "/", alpha.size)
