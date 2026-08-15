#!/usr/bin/env python
"""MVP pitch deck (python-pptx) -- 11 slides, Danilo's structure.

  - slide 8 is an interactive widget: the Arm Control Panel (demo1) rebuilt inside
    PowerPoint as a click-to-play d-pad (real clips extracted from the demo HTML),
    plus a hyperlink to the full browser panel shipped next to the .pptx
  - joke slide (74-sigma "Green Period" blooper, real frames)
  - all numbers from the measured results in CLAUDE.md (256-scene eval, step_8000)

Prerequisites (paths below):
  - widget tiles/posters: `python extract_demo1_widget.py --scene 204`
  - joke frames: last frame of logs/generated_videos/gen_idx100_action-right.mp4
    (the 74-sigma attempt) and logs/gen_4actions/v2_idx100_right.mp4 (the fix),
    nearest-upscaled to 512 and saved as joke/{sludge,fixed}.png
  - full panel copy: cp logs/demo/index.html /home/mls10/presentation/arm_control_panel.html
    (the shipped copy is restyled to the deck template; see CLAUDE.md)

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
JOKE = "/home/mls10/presentation/joke"
OUT_PATH = "/home/mls10/presentation/BAIR_LoRA_Presentation.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
TOTAL = 11


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


def spec_row(slide, top, key, value, sub, width=Inches(10.4)):
    left = Inches(0.9)
    rect(slide, left, top - Inches(0.18), width, Pt(1), LINE)
    add_text(slide, left, top + Inches(0.05), Inches(2.5), Inches(0.4),
             key.upper(), 13, INK_FAINT, font=MONO, letter_spacing=70)
    add_text(slide, left + Inches(2.7), top - Inches(0.06),
             width - Inches(2.7), Inches(0.55), value, 23, INK, bold=True)
    add_text(slide, left + Inches(2.7), top + Inches(0.46),
             width - Inches(2.7), Inches(0.4), sub, 15, INK_DIM, font=MONO)


def chip(slide, left, top, width, height, text, size=15):
    rect(slide, left, top, width, height, BG_RAISED, radius=True)
    rect(slide, left, top, Inches(0.07), height, ACCENT)
    add_text(slide, left + Inches(0.26), top, width - Inches(0.34), height,
             text, size, INK, font=MONO, anchor=MSO_ANCHOR.MIDDLE)


def stat_tile(slide, left, top, width, height, number, label,
              num_color=INK, num_size=28):
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

# ======================================================= 1 · Motivation ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "Motivation")
add_text(s, Inches(0.9), Inches(1.1), Inches(11), Inches(1.6),
         "A robot that can imagine what happens next\ncan plan before it acts.",
         33, INK, bold=True, line_spacing=1.12)

fx, fy, fh = Inches(0.9), Inches(3.15), Inches(0.95)
flow_box(s, fx, fy + Inches(0.55), Inches(1.8), fh, "frame", sub="now")
flow_arrow(s, fx + Inches(1.85), fy + Inches(0.72))
flow_box(s, fx + Inches(2.35), fy + Inches(0.55), Inches(1.7), fh, "model",
         accent=True)
flow_arrow(s, fx + Inches(4.1), fy + Inches(0.72))
chip(s, fx + Inches(4.6), fy, Inches(2.6), Inches(0.58), "←  action: left", 16)
chip(s, fx + Inches(4.6), fy + Inches(0.73), Inches(2.6), Inches(0.58), "↑  action: up", 16)
chip(s, fx + Inches(4.6), fy + Inches(1.46), Inches(2.6), Inches(0.58), "→  action: right", 16)

add_text(s, Inches(0.9), Inches(5.55), Inches(11.2), Inches(1.2),
         "Video models already predict the next frame. The open question:\ncan that prediction be steered by an action, not just left to drift?",
         21, INK_DIM, line_spacing=1.3)
footer(s, 1)

# ============================================================= 2 · Goal ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "Goal")
add_text(s, Inches(0.9), Inches(1.1), Inches(11), Inches(1.5),
         "Fine-tune a pretrained video model\nto obey an action vector.",
         33, INK, bold=True, line_spacing=1.12)

top = Inches(3.35)
for key, val, sub in [
    ("Approach", "LoRA adaptation", "lightweight — the backbone stays frozen"),
    ("Signal", "Real robot actions", "BAIR pushing · recorded end-effector deltas"),
    ("Scope", "Short-horizon rollout", "a demo of the mechanism, not a general planner"),
]:
    spec_row(s, top, key, val, sub)
    top += Inches(1.12)
footer(s, 2)

# ============================================= 3 · Method: model & data ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "Method — Model & Data")
add_text(s, Inches(0.9), Inches(1.1), Inches(11), Inches(1.5),
         "One backbone, one dataset,\none small adapter.",
         33, INK, bold=True, line_spacing=1.12)

top = Inches(3.35)
for key, val, sub in [
    ("Backbone", "Wan2.1-T2V-1.3B", "teacher-forcing checkpoint, before distillation"),
    ("Dataset", "BAIR robot pushing", "64×64 · 43k training clips · 4D action per step"),
    ("Adapter", "LoRA, rank 16", "19M trainable params · q, k, v, ffn.0, ffn.2"),
]:
    spec_row(s, top, key, val, sub)
    top += Inches(1.12)
footer(s, 3)

# ======================================= 4 · Method: action conditioning ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "Method — How the Action Enters")
add_text(s, Inches(0.9), Inches(1.1), Inches(11.5), Inches(1.4),
         "Injected into the per-frame timing signal\n— not the text prompt.",
         31, INK, bold=True, line_spacing=1.12)

fx, fy, fh, fw = Inches(0.9), Inches(3.35), Inches(1.25), Inches(2.55)
gap = Inches(0.45)
flow_box(s, fx, fy, fw, fh, "action", sub="Δx, Δy, …", title_size=17, sub_size=13)
flow_arrow(s, fx + fw, fy + Inches(0.35))
x2 = fx + fw + gap
flow_box(s, x2, fy, fw, fh, "small MLP", sub="16→256→256→1536", title_size=17, sub_size=13)
flow_arrow(s, x2 + fw, fy + Inches(0.35))
x3 = x2 + fw + gap
flow_box(s, x3, fy, fw, fh, "+ timestep\nembedding", accent=True, title_size=17)
flow_arrow(s, x3 + fw, fy + Inches(0.35))
x4 = x3 + fw + gap
flow_box(s, x4, fy, fw, fh, "DiT blocks", sub="+ LoRA, rank 16", title_size=17, sub_size=13)

add_text(s, Inches(0.9), Inches(5.15), Inches(0.4), Inches(0.5), "—", 18, ACCENT, bold=True)
add_text(s, Inches(1.35), Inches(5.15), Inches(10.8), Inches(0.65),
         "Zero-initialized: at step 0 the model is exactly the pretrained checkpoint — nothing breaks.",
         17, INK_DIM, line_spacing=1.25)
add_text(s, Inches(0.9), Inches(5.85), Inches(0.4), Inches(0.5), "—", 18, ACCENT, bold=True)
add_text(s, Inches(1.35), Inches(5.85), Inches(10.8), Inches(0.65),
         "Per-frame, per-block: every one of the 30 DiT blocks hears the action on every frame.",
         17, INK_DIM, line_spacing=1.25)
footer(s, 4)

# ===================================================== 5 · Result (WOW) ====
s = new_slide(prs)
eyebrow(s, Inches(0), Inches(1.0), "Result", align=PP_ALIGN.CENTER,
        width=Inches(13.333))
add_text(s, Inches(0), Inches(1.5), Inches(13.333), Inches(2.5),
         "100%", 145, INK, bold=True, font=MONO, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
add_text(s, Inches(1.67), Inches(4.15), Inches(10), Inches(0.85),
         "of generated clips move in the commanded direction.",
         25, INK_DIM, align=PP_ALIGN.CENTER)

cw, ch, cgap = Inches(2.15), Inches(0.62), Inches(0.25)
row_w = cw * 4 + cgap * 3
cx = (SLIDE_W - row_w) / 2
for glyph in ["↑  up  ✓", "↓  down  ✓", "←  left  ✓", "→  right  ✓"]:
    rect(s, cx, Inches(5.15), cw, ch, BG_RAISED, line_color=ACCENT, radius=True)
    add_text(s, cx, Inches(5.15), cw, ch, glyph, 17, ACCENT, bold=True,
             font=MONO, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    cx += cw + cgap

add_text(s, Inches(1.67), Inches(6.15), Inches(10), Inches(0.4),
         "255 / 256 held-out scenes · saturates by step ~1,000 and never drops",
         14.5, INK_FAINT, font=MONO, align=PP_ALIGN.CENTER)
footer(s, 5)

# ======================================================== 6 · Fidelity ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.55), "Results — Fidelity")
add_text(s, Inches(0.9), Inches(1.0), Inches(11.5), Inches(0.7),
         "The action carries real information.", 31, INK, bold=True)

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
stat_tile(s, tx, ty, tw, th, "+5.3 dB", "value of the action", num_color=ACCENT)
stat_tile(s, tx + (tw + tg), ty, tw, th, "0.78", "SSIM")
stat_tile(s, tx + 2 * (tw + tg), ty, tw, th, "11.1", "FID")
stat_tile(s, tx + 3 * (tw + tg), ty, tw, th, "99.6%", "direction accuracy")

add_text(s, Inches(0.9), Inches(6.68), Inches(11.5), Inches(0.35),
         "256 held-out scenes · final checkpoint · fidelity keeps improving long after control has saturated",
         14, INK_FAINT, font=MONO)
footer(s, 6)

# ============================================ 7 · Interactive widget =======
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.5), "Interactive Demo — Arm Control Panel")
add_text(s, Inches(0.9), Inches(0.95), Inches(11.5), Inches(0.6),
         "Click a command. The model imagines the rest.", 27, INK, bold=True)
add_text(s, Inches(0.9), Inches(1.52), Inches(11.5), Inches(0.4),
         "every tile is a real render from the fine-tuned model · held-out scene · native 64 px",
         14, INK_FAINT, font=MONO)

# left: context screen
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

# middle: d-pad cross of click-to-play movies
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

# right: free rollout chain
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
footer(s, 7)

# ==================================================== 8 · Interlude (joke) ==
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "Interlude")
add_text(s, Inches(0.9), Inches(1.1), Inches(11.5), Inches(0.75),
         "Respect the distribution.", 33, INK, bold=True)

add_text(s, Inches(0.9), Inches(2.2), Inches(4.6), Inches(1.9),
         "74σ", 110, WARN, bold=True, font=MONO)
add_text(s, Inches(0.9), Inches(4.25), Inches(4.7), Inches(0.9),
         "how far out of distribution\nour first commanded action was",
         18, INK_DIM, line_spacing=1.3)
add_text(s, Inches(0.9), Inches(5.25), Inches(4.7), Inches(0.45),
         "typical |Δx| ≈ 0.04 · we asked for 3.0", 15, INK_FAINT, font=MONO)

img = Inches(2.55)
ix1, ix2, iy = Inches(6.6), Inches(9.55), Inches(2.25)
for x, path in [(ix1, os.path.join(JOKE, "fixed.png")),
                (ix2, os.path.join(JOKE, "sludge.png"))]:
    rect(s, x - Inches(0.06), iy - Inches(0.06), img + Inches(0.12),
         img + Inches(0.12), BG_RAISED, line_color=LINE, radius=True)
    s.shapes.add_picture(path, x, iy, img, img)
add_text(s, ix1, iy + img + Inches(0.14), img, Inches(0.65),
         "what we expected\naction in distribution", 13.5, INK_DIM, font=MONO,
         align=PP_ALIGN.CENTER, line_spacing=1.25)
add_text(s, ix2, iy + img + Inches(0.14), img, Inches(0.65),
         "what we got\nthe model's Green Period", 13.5, WARN, font=MONO,
         align=PP_ALIGN.CENTER, line_spacing=1.25)

add_text(s, Inches(0.9), Inches(6.35), Inches(11.5), Inches(0.45),
         "Same checkpoint, same code, one number changed. Fixed within the hour — but first, we framed it.",
         16, INK_DIM, italic=True)
footer(s, 8)

# ===================================================== 9 · Limitations ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "Limitations")
add_text(s, Inches(0.9), Inches(1.1), Inches(11.5), Inches(1.4),
         "Control holds inside one predicted block\n— not yet past it.",
         31, INK, bold=True, line_spacing=1.12)

card_w, card_h = Inches(4.3), Inches(2.3)
cy = Inches(3.15)
rect(s, Inches(0.9), cy, card_w, card_h, BG_RAISED, line_color=ACCENT,
     radius=True, line_w=1.75)
add_text(s, Inches(0.9), cy + Inches(0.2), card_w, Inches(1.2), "97%", 68,
         ACCENT, bold=True, font=MONO, align=PP_ALIGN.CENTER)
add_text(s, Inches(1.1), cy + Inches(1.55), card_w - Inches(0.4), Inches(0.6),
         "direction accuracy\nconditioned on real context", 14.5, INK_DIM,
         font=MONO, align=PP_ALIGN.CENTER, line_spacing=1.25)

add_text(s, Inches(5.35), cy + Inches(0.65), Inches(1.0), Inches(1.0), "→",
         40, INK_FAINT, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

rect(s, Inches(6.5), cy, card_w, card_h, BG_RAISED, line_color=WARN,
     radius=True, line_w=1.75)
add_text(s, Inches(6.5), cy + Inches(0.2), card_w, Inches(1.2), "53%", 68,
         WARN, bold=True, font=MONO, align=PP_ALIGN.CENTER)
add_text(s, Inches(6.7), cy + Inches(1.55), card_w - Inches(0.4), Inches(0.6),
         "after one self-generated block\n— chance level", 14.5, INK_DIM,
         font=MONO, align=PP_ALIGN.CENTER, line_spacing=1.25)

add_text(s, Inches(0.9), Inches(5.85), Inches(11.5), Inches(1.1),
         "Trained only on real context (teacher forcing). Fed its own output, quality and control degrade\nblock by block — exposure bias, the known cost of the safer training stage we chose.",
         17, INK_DIM, line_spacing=1.3)
footer(s, 9)

# ============================================================ 10 · Next ====
s = new_slide(prs)
eyebrow(s, Inches(0.9), Inches(0.62), "Next")
add_text(s, Inches(0.9), Inches(1.1), Inches(11.5), Inches(1.4),
         "The mechanism works.\nMaking it hold past one block is next.",
         31, INK, bold=True, line_spacing=1.12)

top = Inches(3.45)
for title, sub in [
    ("Scheduled sampling",
     "train on the model's own imperfect output some of the time,\nso it learns to recover · already running"),
    ("Goal-conditioned action search",
     "the same model, run backwards: given a goal frame, search which\naction gets there — forward ↔ inverse dynamics"),
]:
    add_text(s, Inches(0.9), top, Inches(0.5), Inches(0.5), "→", 24, ACCENT, bold=True)
    add_text(s, Inches(1.55), top - Inches(0.04), Inches(10.5), Inches(0.55),
             title, 24, INK, bold=True)
    add_text(s, Inches(1.55), top + Inches(0.5), Inches(10.5), Inches(0.75),
             sub, 15.5, INK_DIM, font=MONO, line_spacing=1.3)
    top += Inches(1.6)
footer(s, 10)

prs.save(OUT_PATH)
print("saved:", OUT_PATH, f"{os.path.getsize(OUT_PATH)/1e6:.2f} MB")
