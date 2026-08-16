#!/usr/bin/env python
"""Static frames for the inverse-dynamics slide (one search cycle, goal =
the 'right' future). Replaces the auto-playing search_tree.gif: each frame
becomes its own slide in the deck, so the presenter steps through the search
by hand (click to advance) instead of watching a timed loop.

Same real data / same argmin logic as make_search_tree_gif.py, just frozen
into 7 individual PNGs instead of one animated GIF.
"""
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib

FDIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
FONT_B = ImageFont.truetype(os.path.join(FDIR, "DejaVuSans-Bold.ttf"), 34)
FONT_S = ImageFont.truetype(os.path.join(FDIR, "DejaVuSans-Bold.ttf"), 27)

WIDGET = "/home/mls10/presentation/widget"
OUT_DIR = "/home/mls10/presentation/search_steps"
os.makedirs(OUT_DIR, exist_ok=True)

BG = (16, 20, 38)
INK = (246, 247, 251)
DIM = (155, 166, 190)
TEAL = (62, 230, 208)
AMBER = (255, 209, 102)
PANEL = (10, 12, 24)

W, H = 1120, 720


def last_frame(name):
    rdr = imageio.get_reader(os.path.join(WIDGET, f"tile_{name}.mp4"))
    frames = [f for f in rdr]
    return np.asarray(frames[-1])[..., :3]


ctx = np.asarray(Image.open(os.path.join(WIDGET, "screen_context.png")).convert("RGB"))
actions = ["up", "down", "left", "right", "still"]
glyphs = {"up": "↑", "down": "↓", "left": "←", "right": "→", "still": "■"}
futures = {a: last_frame(a) for a in actions}


def score_against(goal_img):
    sc = {a: float(np.mean(np.abs(futures[a].astype(np.float32) - goal_img.astype(np.float32))))
          for a in actions}
    return sc, min(sc, key=sc.get)


# goal = the "up" trajectory's OWN final frame -- picked because it's the
# candidate with the largest visible displacement from "now" (checked all
# five last-frames side by side; "right"/"down"/"still" barely move the arm
# over 16 frames and made the match look coincidental, not found)
goal = futures["up"]
scores, best = score_against(goal)
print("best:", best, {a: round(v, 1) for a, v in scores.items()})

ROOT = (95, H // 2 - 40)
GOAL = (95, 40)
LEAF_X = 780
leaf_ys = [30 + i * 138 for i in range(5)]
THUMB, LEAF, GOALS = 190, 120, 120


def thumb(img, size):
    return Image.fromarray(img).resize((size, size), Image.LANCZOS)


def canvas(revealed, active=None, final=False):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([GOAL[0] - 4, GOAL[1] - 4, GOAL[0] + GOALS + 4, GOAL[1] + GOALS + 4],
                outline=AMBER, width=4)
    im.paste(thumb(goal, GOALS), GOAL)
    d.text((GOAL[0] + 22, GOAL[1] + GOALS + 10), "goal", font=FONT_B, fill=AMBER)
    show = best if final else active
    cx0, cy0 = GOAL[0] + GOALS + 170, GOAL[1]
    d.text((GOAL[0] + GOALS + 60, GOAL[1] + GOALS // 2 - 24), "vs", font=FONT_B, fill=DIM)
    if show is not None:
        c_col = TEAL if final else AMBER
        d.rectangle([cx0 - 4, cy0 - 4, cx0 + GOALS + 4, cy0 + GOALS + 4],
                    outline=c_col, width=4)
        im.paste(thumb(futures[show], GOALS), (cx0, cy0))
        d.text((cx0 - 14, cy0 + GOALS + 10),
               f"imagined {glyphs[show]}", font=FONT_B, fill=c_col)
    else:
        d.rectangle([cx0 - 4, cy0 - 4, cx0 + GOALS + 4, cy0 + GOALS + 4],
                    outline=(70, 78, 98), width=4)
        d.text((cx0 - 14, cy0 + GOALS + 10), "imagined ?", font=FONT_B, fill=DIM)
    ry = ROOT[1]
    d.rectangle([ROOT[0] - 4, ry - 4, ROOT[0] + THUMB + 4, ry + THUMB + 4],
                outline=(111, 211, 255), width=4)
    im.paste(thumb(ctx, THUMB), (ROOT[0], ry))
    d.text((ROOT[0], ry + THUMB + 12), "now", font=FONT_B, fill=(111, 211, 255))
    rcx, rcy = ROOT[0] + THUMB, ry + THUMB // 2

    for i, a in enumerate(actions):
        ly = leaf_ys[i]
        lcy = ly + LEAF // 2
        is_best = (a == best)
        if final and is_best:
            col, wdt = TEAL, 8
        elif a == active:
            col, wdt = AMBER, 6
        else:
            col, wdt = DIM, 3
        d.line([rcx + 8, rcy, LEAF_X - 64, lcy], fill=col, width=wdt)
        gx, gy = (rcx + LEAF_X - 64) // 2, (rcy + lcy) // 2
        d.ellipse([gx - 26, gy - 26, gx + 26, gy + 26], fill=PANEL, outline=col, width=3)
        d.text((gx - 11, gy - 21), glyphs[a], font=FONT_B, fill=INK)
        if a in revealed or a == active:
            if final and is_best:
                box_col, bw = TEAL, 6
            elif a == active:
                box_col, bw = AMBER, 5
            else:
                box_col, bw = (90, 99, 120), 3
            d.rectangle([LEAF_X - 4, ly - 4, LEAF_X + LEAF + 4, ly + LEAF + 4],
                        outline=box_col, width=bw)
            im.paste(thumb(futures[a], LEAF), (LEAF_X, ly))
            s_col = TEAL if (final and is_best) else (AMBER if a == active else DIM)
            d.text((LEAF_X + LEAF + 20, ly + LEAF // 2 - 32), "L1", font=FONT_S, fill=s_col)
            d.text((LEAF_X + LEAF + 20, ly + LEAF // 2 - 2), f"{scores[a]:.0f}",
                   font=FONT_B, fill=s_col)
    if final:
        d.text((240, H - 52), f"pick:  {glyphs[best]}  (closest to goal)",
               font=FONT_B, fill=TEAL)
    return im


steps = []
steps.append(canvas(set()))                       # 0: setup, nothing revealed
seen = []
for a in actions:                                  # 1-5: reveal one branch per click
    steps.append(canvas(set(seen), active=a))
    seen.append(a)
steps.append(canvas(set(seen), final=True))         # 6: winner highlighted

for i, im in enumerate(steps):
    path = os.path.join(OUT_DIR, f"step{i}.png")
    im.save(path)
    print("saved", path)
