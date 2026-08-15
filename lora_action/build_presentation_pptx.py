#!/usr/bin/env python
"""Pitch deck builder (python-pptx) -- title + 10 content slides, per PREZENTACIJA_BRIEF.md.

Structure follows the brief (10 min + 5 Q&A): motivation+goals, method (2 slides:
the two questions the audience will ask), control result (99.6% -- NOT the 100%
subset number, per the brief's mandatory caveat), the controllability ladder,
the interactive widget, inverse dynamics, limitations (the
corrected story: control survives rollout, localization drifts), and a closing
slide with the four measured contributions.

Interactive widget: PowerPoint cannot embed HTML, so demo1's frames are extracted
from its base64 DATA blob (extract_demo1_widget.py --scene 204) and rebuilt as
click-to-play movie tiles, with a relative hyperlink to the full panel shipped
next to the .pptx (arm_control_panel.html -- restyled to this deck's template).

Prerequisites:
  - widget tiles/posters in /home/mls10/presentation/widget (extract_demo1_widget.py)
  - goal-search panel: logs/goal_search/goal_search_idx108.png
  - cp logs/demo/index.html /home/mls10/presentation/arm_control_panel.html

Gotchas that cost time: python-pptx 1.0.2 has no `pptx.enum.line` (dash style lives
in `pptx.enum.dml.MSO_LINE_DASH_STYLE`), and a textbox overlaid on a hyperlinked
shape intercepts the click -- the link text must live in the shape's own text_frame.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE as MSO_LINE

# ---------------------------------------------------------------- palette --
BG        = RGBColor(0x0A, 0x0C, 0x0F)
BG_RAISED = RGBColor(0x14, 0x18, 0x1E)
BG_INSET  = RGBColor(0x1A, 0x1E, 0x25)
INK       = RGBColor(0xF2, 0xF4, 0xF6)
INK_DIM   = RGBColor(0xA6, 0xAF, 0xBA)
INK_FAINT = RGBColor(0x6E, 0x77, 0x82)
ACCENT    = RGBColor(0x38, 0xD6, 0xC4)
WARN      = RGBColor(0xFF, 0x7A, 0x52)
LINE      = RGBColor(0x2C, 0x31, 0x39)

SANS = "Calibri"
MONO = "Consolas"

WIDGET = "/home/mls10/presentation/widget"
GOAL_PANEL = "/home/mls10/logs/goal_search/goal_search_idx108.png"
OUT_PATH = "/home/mls10/presentation/BAIR_LoRA_Presentation.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
TOTAL = 10


# ------------------------------------------------------------- utilities --
def set_letter_spacing(run, pt_hundredths):
    rPr = run._r.get_or_add_rPr()
    rPr.set("spc", str(pt_hundredths))


def new_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    return slide


def add_text(slide, left, top, width, height, text, size, color,
             bold=False, italic=False, font=SANS, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, line_spacing=1.12, letter_spacing=None,
             upper=False):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line.upper() if upper else line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.name = font
        run.font.color.rgb = color
        if letter_spacing is not None:
            set_letter_spacing(run, letter_spacing)
    return tb


def eyebrow(slide, left, top, text, align=PP_ALIGN.LEFT, width=Inches(11)):
    add_text(slide, left, top, width, Inches(0.4), text, 14, ACCENT,
             bold=True, font=MONO, align=align, letter_spacing=150, upper=True)


def footer(slide, index):
    add_text(slide, Inches(0.9), Inches(7.08), Inches(7), Inches(0.32),
             "BAIR ROBOT PUSHING  ·  WAN2.1-T2V-1.3B  ·  LORA RANK 16",
             10.5, INK_FAINT, font=MONO, letter_spacing=60)
    add_text(slide, Inches(11.2), Inches(7.08), Inches(1.3), Inches(0.32),
             f"{index+1:02d} / {TOTAL:02d}", 10.5, INK_FAINT, font=MONO,
             align=PP_ALIGN.RIGHT)


def rect(slide, left, top, width, height, fill_color, line_color=None,
         radius=False, line_w=1.25):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_color
    if line_color is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    if radius:
        try:
            shp.adjustments[0] = 0.10
        except Exception:
            pass
    return shp


def flow_box(slide, left, top, width, height, title, sub=None, accent=False,
             title_size=16, sub_size=12):
    rect(slide, left, top, width, height, BG_RAISED,
         line_color=ACCENT if accent else LINE, radius=True,
         line_w=1.75 if accent else 1.25)
    if sub:
        add_text(slide, left + Inches(0.08), top + Inches(0.16),
                 width - Inches(0.16), Inches(0.5), title, title_size, INK,
                 bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.BOTTOM)
        add_text(slide, left + Inches(0.08), top + height - Inches(0.52),
                 width - Inches(0.16), Inches(0.44), sub, sub_size, INK_DIM,
                 font=MONO, align=PP_ALIGN.CENTER)
    else:
        add_text(slide, left + Inches(0.08), top, width - Inches(0.16), height,
                 title, title_size, INK, bold=True, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE)


def flow_arrow(slide, left, top):
    add_text(slide, left, top, Inches(0.45), Inches(0.6), "→", 24, INK_FAINT,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def spec_row(slide, top, key, value, sub, width=Inches(10.6)):
    left = Inches(0.9)
    rect(slide, left, top - Inches(0.16), width, Pt(1), LINE)
    add_text(slide, left, top + Inches(0.05), Inches(2.3), Inches(0.4),
             key.upper(), 13, INK_FAINT, font=MONO, letter_spacing=70)
    add_text(slide, left + Inches(2.5), top - Inches(0.06),
             width - Inches(2.5), Inches(0.55), value, 22, INK, bold=True)
    add_text(slide, left + Inches(2.5), top + Inches(0.42),
             width - Inches(2.5), Inches(0.4), sub, 14.5, INK_DIM, font=MONO)


def chip(slide, left, top, width, height, text, size=15):
    rect(slide, left, top, width, height, BG_RAISED, radius=True)
    rect(slide, left, top, Inches(0.07), height, ACCENT)
    add_text(slide, left + Inches(0.26), top, width - Inches(0.34), height,
             text, size, INK, font=MONO, anchor=MSO_ANCHOR.MIDDLE)


def stat_tile(slide, left, top, width, height, number, label,
              num_color=INK, num_size=27):
    rect(slide, left, top, width, height, BG_RAISED, line_color=LINE, radius=True)
    add_text(slide, left, top + Inches(0.1), width, Inches(0.55), number,
             num_size, num_color, bold=True, font=MONO, align=PP_ALIGN.CENTER)
    add_text(slide, left + Inches(0.1), top + height - Inches(0.42),
             width - Inches(0.2), Inches(0.36), label, 12.5, INK_FAINT,
             font=MONO, align=PP_ALIGN.CENTER, letter_spacing=40, upper=True)


# ------------------------------------------------------------------ build --
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

# ============================================================ 0 · Title ====
s = new_slide(prs)
add_text(s, Inches(1), Inches(2.0), Inches(11.33), Inches(0.4),
         "BAIR ROBOT PUSHING  ·  WAN2.1-T2V-1.3B  ·  LORA", 13.5,
         INK_FAINT, font=MONO, align=PP_ALIGN.CENTER, letter_spacing=150)
add_text(s, Inches(1), Inches(2.55), Inches(11.33), Inches(1.9),
         "Action-Conditioned\nWorld Models", 46, INK, bold=True,
         align=PP_ALIGN.CENTER, line_spacing=1.05)
rect(s, Inches(5.57), Inches(4.42), Inches(2.2), Pt(3), ACCENT)
add_text(s, Inches(2.17), Inches(4.75), Inches(9), Inches(0.9),
         "Teaching a pretrained video diffusion model to obey a command,\nnot just guess the future.",
         21, INK_DIM, align=PP_ALIGN.CENTER, line_spacing=1.28)
add_text(s, Inches(1), Inches(5.85), Inches(11.33), Inches(0.4),
         "Dawidzard & Mihajlo   ·   advised by Nedko Savov & Danilo Djordjevic, INSAIT",
         14.5, INK_FAINT, font=MONO, align=PP_ALIGN.CENTER)

# ================================================ 1 · Motivation & goals ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "Motivation")
add_text(s, Inches(0.9), Inches(1.1), Inches(11), Inches(1.6),
         "A robot that can imagine what happens next\ncan plan before it acts.",
         33, INK, bold=True, line_spacing=1.12)

fx, fy, fh = Inches(0.9), Inches(2.95), Inches(0.95)
flow_box(s, fx, fy + Inches(0.55), Inches(1.8), fh, "frame", sub="now")
flow_arrow(s, fx + Inches(1.85), fy + Inches(0.72))
flow_box(s, fx + Inches(2.35), fy + Inches(0.55), Inches(1.7), fh, "model",
         accent=True)
flow_arrow(s, fx + Inches(4.1), fy + Inches(0.72))
chip(s, fx + Inches(4.6), fy, Inches(2.6), Inches(0.58), "←  action: left", 16)
chip(s, fx + Inches(4.6), fy + Inches(0.73), Inches(2.6), Inches(0.58), "↑  action: up", 16)
chip(s, fx + Inches(4.6), fy + Inches(1.46), Inches(2.6), Inches(0.58), "→  action: right", 16)
add_text(s, fx + Inches(7.6), fy + Inches(0.55), Inches(3.9), Inches(1.5),
         "Can a pretrained video model be made to obey a robot command —\nand how precisely?",
         19, INK_DIM, line_spacing=1.3, anchor=MSO_ANCHOR.MIDDLE)

add_text(s, Inches(0.9), Inches(5.45), Inches(1.4), Inches(0.5), "GOALS", 13,
         INK_FAINT, font=MONO, letter_spacing=80)
gw = Inches(3.65)
gg = Inches(0.2)
for i, g in enumerate(["condition on real robot actions",
                       "measure how precisely it obeys",
                       "find where it fails, and why"]):
    chip(s, Inches(0.9) + i * (gw + gg), Inches(5.85), gw, Inches(0.6), g, 14)
footer(s, 1)

# ============================================= 2 · Method: model & data ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "Method — Model & Data")
add_text(s, Inches(0.9), Inches(1.1), Inches(11), Inches(1.4),
         "One backbone, one dataset,\none small adapter.",
         33, INK, bold=True, line_spacing=1.12)

top = Inches(3.0)
for key, val, sub in [
    ("Backbone", "Wan2.1-T2V-1.3B", "causal autoregressive video model (minWM) · teacher-forcing checkpoint"),
    ("In / Out", "4 latent frames + action → next 4", "= 16 pixel frames per generated block"),
    ("Dataset", "BAIR robot pushing, 64×64", "216,325 training windows · 256 held-out test scenes · 4D actions"),
    ("Adapter", "LoRA r16 + action encoder", "19.4M trainable = 1.4% of the model · 15 lines changed upstream"),
]:
    spec_row(s, top, key, val, sub)
    top += Inches(1.0)
footer(s, 2)

# ======================================= 3 · Method: the two questions ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "Method — Action Conditioning")
add_text(s, Inches(0.9), Inches(1.1), Inches(11.5), Inches(0.7),
         "The two questions everyone asks.", 33, INK, bold=True)

fx, fy, fh, fw = Inches(0.9), Inches(2.15), Inches(1.2), Inches(2.55)
gap = Inches(0.45)
flow_box(s, fx, fy, fw, fh, "action", sub="4 per frame, flattened", title_size=17, sub_size=12)
flow_arrow(s, fx + fw, fy + Inches(0.32))
x2 = fx + fw + gap
flow_box(s, x2, fy, fw, fh, "small MLP", sub="16→256→256→1536", title_size=17, sub_size=12)
flow_arrow(s, x2 + fw, fy + Inches(0.32))
x3 = x2 + fw + gap
flow_box(s, x3, fy, fw, fh, "+ timestep\nembedding", accent=True, title_size=17)
flow_arrow(s, x3 + fw, fy + Inches(0.32))
x4 = x3 + fw + gap
flow_box(s, x4, fy, fw, fh, "AdaLN, all\n30 DiT blocks", sub="+ LoRA r16", title_size=16, sub_size=12)

add_text(s, Inches(0.9), Inches(3.85), Inches(11.5), Inches(0.45),
         "A — How are the action embeddings built?", 19, INK, bold=True)
add_text(s, Inches(0.9), Inches(4.3), Inches(11.5), Inches(0.95),
         "Per latent frame: its 4 raw actions, flattened — never averaged. Actions are deltas, so the mean\nof “left, then back” is zero and the signal dies. Zero-init output layer: step 0 = untouched pretrained model.",
         15.5, INK_DIM, font=MONO, line_spacing=1.3)

add_text(s, Inches(0.9), Inches(5.45), Inches(11.5), Inches(0.45),
         "B — How does the action steer the video?", 19, INK, bold=True)
add_text(s, Inches(0.9), Inches(5.9), Inches(11.5), Inches(0.95),
         "Added to the per-frame timestep embedding → AdaLN shift/scale/gate. The action modulates every\nblock on every frame — the text prompt is fixed and carries nothing.",
         15.5, INK_DIM, font=MONO, line_spacing=1.3)
footer(s, 3)

# ================================================= 4 · Result: control ====
s = new_slide(prs)
eyebrow(s, Inches(0), Inches(1.0), "Result", align=PP_ALIGN.CENTER,
        width=Inches(13.333))
add_text(s, Inches(0), Inches(1.45), Inches(13.333), Inches(2.4),
         "99.6%", 135, INK, bold=True, font=MONO, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
add_text(s, Inches(1.67), Inches(4.0), Inches(10), Inches(0.85),
         "of held-out scenes move in the commanded direction — 255 of 256.",
         24, INK_DIM, align=PP_ALIGN.CENTER)

cw, ch, cgap = Inches(2.15), Inches(0.62), Inches(0.25)
row_w = cw * 4 + cgap * 3
cx = (SLIDE_W - row_w) / 2
for glyph in ["↑  up  ✓", "↓  down  ✓", "←  left  ✓", "→  right  ✓"]:
    rect(s, cx, Inches(5.0), cw, ch, BG_RAISED, line_color=ACCENT, radius=True)
    add_text(s, cx, Inches(5.0), cw, ch, glyph, 17, ACCENT, bold=True,
             font=MONO, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    cx += cw + cgap

add_text(s, Inches(1.17), Inches(5.95), Inches(11), Inches(0.75),
         "absolute direction accuracy 84.8% — the gap is scene dynamics carrying the arm, not disobedience\ncontrol saturates by ~32,000 samples and never drops · untrained baseline: 51.5% = chance",
         13.5, INK_FAINT, font=MONO, align=PP_ALIGN.CENTER, line_spacing=1.35)
footer(s, 4)

# ============================================ 5 · Controllability ladder ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.55), "Results — What the Action Is Worth")
add_text(s, Inches(0.9), Inches(1.0), Inches(11.5), Inches(0.7),
         "Same model, four levels of being informed.", 31, INK, bold=True)

chart_left = Inches(3.7)
chart_width = Inches(7.3)
ceiling_db = 22.74
scale_max = 24.0
bars = [
    ("no fine-tune", 7.12, False),
    ("wrong action", 12.45, False),
    ("no action (null)", 13.27, False),
    ("real action", 18.56, True),
]
row_h = Inches(0.52)
row_gap = Inches(0.30)
top = Inches(2.15)
for label, val, is_real in bars:
    add_text(s, Inches(0.9), top, Inches(2.7), row_h, label, 16, INK_DIM,
             font=MONO, anchor=MSO_ANCHOR.MIDDLE)
    rect(s, chart_left, top, chart_width, row_h, BG_INSET, radius=True)
    fw_bar = Emu(int(chart_width * (val / scale_max)))
    rect(s, chart_left, top, fw_bar, row_h,
         ACCENT if is_real else INK_FAINT, radius=True)
    add_text(s, chart_left + chart_width + Inches(0.15), top, Inches(1.3),
             row_h, f"{val:.1f} dB", 18, INK, bold=True, font=MONO,
             anchor=MSO_ANCHOR.MIDDLE)
    top += row_h + row_gap

chart_bottom = top - row_gap + Inches(0.05)
ceiling_x = chart_left + Emu(int(chart_width * (ceiling_db / scale_max)))
conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, ceiling_x, Inches(1.85),
                              ceiling_x, chart_bottom)
conn.line.color.rgb = WARN
conn.line.width = Pt(1.75)
conn.line.dash_style = MSO_LINE.DASH
add_text(s, ceiling_x - Inches(2.6), Inches(1.52), Inches(2.5), Inches(0.32),
         "VAE ceiling · 22.7 dB", 13, WARN, font=MONO, align=PP_ALIGN.RIGHT)

tw, th, tg = Inches(2.72), Inches(1.05), Inches(0.22)
tx = Inches(0.9)
ty = Inches(5.5)
stat_tile(s, tx, ty, tw, th, "+5.29 dB", "value of the action", num_color=ACCENT)
stat_tile(s, tx + (tw + tg), ty, tw, th, "0.785", "SSIM")
stat_tile(s, tx + 2 * (tw + tg), ty, tw, th, "11.1", "FID")
stat_tile(s, tx + 3 * (tw + tg), ty, tw, th, "81%", "of the VAE ceiling")

add_text(s, Inches(0.9), Inches(6.68), Inches(11.5), Inches(0.35),
         "PSNR, 256 held-out scenes, final checkpoint · untrained: FID 229, ΔPSNR −0.03 — none of this was already in the model",
         13.5, INK_FAINT, font=MONO)
footer(s, 5)

# ============================================ 6 · Interactive widget =======
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.5), "Interactive Demo — Arm Control Panel")
add_text(s, Inches(0.9), Inches(0.95), Inches(11.5), Inches(0.6),
         "Click a command. The model imagines the rest.", 27, INK, bold=True)
add_text(s, Inches(0.9), Inches(1.52), Inches(11.5), Inches(0.4),
         "real renders · held-out scene · native 64 px · 4 denoising steps (24 steps buys 12% FID for 6× compute)",
         13.5, INK_FAINT, font=MONO)

scr = Inches(2.95)
sx, sy = Inches(0.9), Inches(2.15)
rect(s, sx - Inches(0.1), sy - Inches(0.1), scr + Inches(0.2), scr + Inches(0.2),
     BG_RAISED, line_color=LINE, radius=True)
s.shapes.add_picture(os.path.join(WIDGET, "screen_context.png"), sx, sy, scr, scr)
add_text(s, sx, sy + scr + Inches(0.18), scr, Inches(0.35),
         "START FRAME", 13, INK_FAINT, font=MONO, align=PP_ALIGN.CENTER,
         letter_spacing=80)

link = rect(s, sx, sy + scr + Inches(0.62), scr, Inches(0.55), BG_RAISED,
            line_color=ACCENT, radius=True)
ltf = link.text_frame
ltf.word_wrap = True
ltf.vertical_anchor = MSO_ANCHOR.MIDDLE
lp = ltf.paragraphs[0]
lp.alignment = PP_ALIGN.CENTER
lr = lp.add_run()
lr.text = "OPEN FULL PANEL ↗"
lr.font.size = Pt(14.5)
lr.font.bold = True
lr.font.name = MONO
lr.font.color.rgb = ACCENT
set_letter_spacing(lr, 60)
link.click_action.hyperlink.address = "arm_control_panel.html"

tile = Inches(1.55)
tgap = Inches(0.13)
px = Inches(4.85)
py = Inches(2.0)
cols = [px, px + tile + tgap, px + 2 * (tile + tgap)]
rows = [py, py + tile + tgap, py + 2 * (tile + tgap)]
pad = [("up", cols[1], rows[0]), ("left", cols[0], rows[1]),
       ("still", cols[1], rows[1]), ("right", cols[2], rows[1]),
       ("down", cols[1], rows[2])]
for name, x, y in pad:
    s.shapes.add_movie(os.path.join(WIDGET, f"tile_{name}.mp4"), x, y, tile, tile,
                       poster_frame_image=os.path.join(WIDGET, f"tile_{name}.png"),
                       mime_type="video/mp4")

chx, chy, chs = Inches(10.35), Inches(2.35), Inches(2.1)
s.shapes.add_movie(os.path.join(WIDGET, "tile_chain.mp4"), chx, chy, chs, chs,
                   poster_frame_image=os.path.join(WIDGET, "tile_chain.png"),
                   mime_type="video/mp4")
add_text(s, chx - Inches(0.2), chy + chs + Inches(0.14), chs + Inches(0.4),
         Inches(0.35), "FREE ROLLOUT ×4", 13, INK_FAINT, font=MONO,
         align=PP_ALIGN.CENTER, letter_spacing=80)
add_text(s, chx - Inches(0.35), chy + chs + Inches(0.52), chs + Inches(0.7),
         Inches(0.6), "the model feeds on\nits own output — watch it drift",
         12.5, INK_FAINT, font=MONO, align=PP_ALIGN.CENTER, line_spacing=1.25)
footer(s, 6)

# ============================================= 7 · Inverse dynamics ========
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.5), "Results — Inverse Dynamics")
add_text(s, Inches(0.9), Inches(0.95), Inches(11.5), Inches(0.65),
         "Choosing an action by imagining its outcome.", 29, INK, bold=True)

panel_w = Inches(10.4)
panel_h = Inches(2.6)   # source is 768x192 = 4:1
panel_x = Inches(1.47)
panel_y = Inches(1.85)
rect(s, panel_x - Inches(0.06), panel_y - Inches(0.06), panel_w + Inches(0.12),
     panel_h + Inches(0.12), BG_RAISED, line_color=LINE, radius=True)
s.shapes.add_picture(GOAL_PANEL, panel_x, panel_y, panel_w, panel_h)
q = panel_w / 4
for i, cap in enumerate(["GOAL (real future)", "BEST imagined", "WORST imagined",
                         "score over (dx,dy) · white = closer"]):
    add_text(s, panel_x + i * q, panel_y + panel_h + Inches(0.12), q,
             Inches(0.35), cap, 12, INK_FAINT, font=MONO, align=PP_ALIGN.CENTER)

stat_tile(s, Inches(0.9), Inches(5.15), Inches(2.5), Inches(1.05), "15 / 20",
          "sign agreement", num_color=ACCENT)
add_text(s, Inches(3.7), Inches(5.15), Inches(8.7), Inches(1.15),
         "36 candidate actions, one batched pass, same noise — pick the imagined future closest to the goal.\nChance is 10/20. This scene: chose (+0.042, +0.014), truth was (+0.035, +0.012) — direction and\nmagnitude order right, without ever seeing the answer. Bias: it overshoots strength.",
         14, INK_DIM, font=MONO, line_spacing=1.3)
add_text(s, Inches(0.9), Inches(6.55), Inches(11.5), Inches(0.35),
         "misses cluster where the true action ≈ 0, where sign is meaningless · the goal is the episode's own recorded future — verifiable, not “looks plausible”",
         12.5, INK_FAINT, font=MONO)
footer(s, 7)

# ===================================================== 8 · Limitations ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.55), "Limitations — Free Rollout")
add_text(s, Inches(0.9), Inches(1.0), Inches(11.6), Inches(1.35),
         "Two different things fail — and only one\nis the one we suspected.",
         30, INK, bold=True, line_spacing=1.12)

card_w, card_h = Inches(3.72), Inches(2.35)
cgap = Inches(0.25)
cy = Inches(2.75)
cards = [
    ("98–100%", ACCENT, "relative control at depth 3\nthe model never stops obeying"),
    ("79% → 72%", WARN, "absolute accuracy, depth 1→2\nthe scene drifts under the arm"),
    ("+97% → +57%", INK, "image decay (FID), halved by\nscheduled sampling · −1.5 dB cost"),
]
for i, (num, color, label) in enumerate(cards):
    x = Inches(0.9) + i * (card_w + cgap)
    rect(s, x, cy, card_w, card_h, BG_RAISED,
         line_color=color if color != INK else LINE, radius=True, line_w=1.75)
    add_text(s, x, cy + Inches(0.28), card_w, Inches(0.9), num, 40, color,
             bold=True, font=MONO, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.2), cy + Inches(1.4), card_w - Inches(0.4),
             Inches(0.85), label, 13.5, INK_DIM, font=MONO,
             align=PP_ALIGN.CENTER, line_spacing=1.3)

add_text(s, Inches(0.9), Inches(5.5), Inches(11.6), Inches(1.3),
         "Exposure bias explains the image decay — not the localization drift. So the fix is scene anchoring\nand longer context, not stronger conditioning. Scheduled sampling (self-predicted context on 50% of\nsamples) halves the decay but costs one-step fidelity: a trade, not a win. All numbers: 256 tries.",
         15.5, INK_DIM, line_spacing=1.32)
footer(s, 8)

# ============================================ 9 · Contributions & next ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "What We Actually Measured")
add_text(s, Inches(0.9), Inches(1.1), Inches(11.5), Inches(0.7),
         "Four findings a single number would hide.", 31, INK, bold=True)

items = [
    ("Control and fidelity mature on different time scales",
     "control saturates at ~32k samples · fidelity keeps improving 8× longer"),
    ("Rollout failure is two separable failures",
     "exposure bias degrades the image · localization drift loses the arm"),
    ("Distillation isn't needed here — a negative result",
     "the base doesn't matter, and 4-step sampling survives fine-tuning: 6× faster, free"),
    ("The VAE sets the ceiling: 22.74 dB",
     "our 18.56 dB is 81% of what 64×64 allows — context for every PSNR"),
]
top = Inches(2.1)
for i, (title, sub) in enumerate(items):
    add_text(s, Inches(0.9), top, Inches(0.55), Inches(0.5), f"{i+1}", 22,
             ACCENT, bold=True, font=MONO)
    add_text(s, Inches(1.55), top - Inches(0.03), Inches(10.6), Inches(0.5),
             title, 19.5, INK, bold=True)
    add_text(s, Inches(1.55), top + Inches(0.4), Inches(10.6), Inches(0.4),
             sub, 13.5, INK_DIM, font=MONO)
    top += Inches(0.98)

add_text(s, Inches(0.9), Inches(6.1), Inches(11.6), Inches(0.4),
         "Two of the four are negative. All four we could have kept quiet.",
         17, INK, italic=True)
add_text(s, Inches(0.9), Inches(6.62), Inches(11.6), Inches(0.35),
         "next: scene anchoring · real self-forcing · 256×256 (lifts the ceiling) · action-strength calibration",
         13.5, INK_FAINT, font=MONO)
footer(s, 9)

prs.save(OUT_PATH)
print("saved:", OUT_PATH, f"{os.path.getsize(OUT_PATH)/1e6:.2f} MB")
