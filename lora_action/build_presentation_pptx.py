#!/usr/bin/env python
"""Pitch deck builder (python-pptx) -- title + 9 content slides, per PREZENTACIJA_BRIEF.md.

Visual redesign round (user request, day 5):
  - Times New Roman everywhere, no text below 30 pt (small captions reformulated
    or cut -- the speakers narrate them), footers removed
  - per-slide two-stop gradient backgrounds, multi-color accents, gradient bars
  - title slide without mentors
  - slide 3 is a full-slide implementation-architecture diagram (no headline):
    video lane and action lane converge into the DiT, output decoded below
  - slide 4 reduced to the two design decisions, one card each
  - slide 6 ladder rebuilt label-above-bar (nothing side-by-side can overlap),
    one color per rung

Content still follows the brief: 99.6% (not the 100% subset), the ladder,
the click-to-play widget (demo1, scene 204), inverse dynamics 15/20, the
corrected rollout story (control survives, localization drifts), four
contributions. Numbers all from the 256-scene eval unless said otherwise.

Prerequisites:
  - widget tiles/posters in /home/mls10/presentation/widget (extract_demo1_widget.py)
  - goal-search panel: logs/goal_search/goal_search_idx108.png
  - cp logs/demo/index.html /home/mls10/presentation/arm_control_panel.html

Gotchas: python-pptx 1.0.2 dash style lives in pptx.enum.dml.MSO_LINE_DASH_STYLE;
hyperlink text must live in the linked shape's own text_frame; connector
arrowheads need raw oxml (a:tailEnd).
"""
import copy
import os
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE as MSO_LINE
from pptx.oxml.ns import qn

# ---------------------------------------------------------------- palette --
INK    = RGBColor(0xF6, 0xF7, 0xFB)
DIM    = RGBColor(0xCB, 0xD3, 0xE4)
PANEL  = RGBColor(0x10, 0x14, 0x26)
TEAL   = RGBColor(0x3E, 0xE6, 0xD0)
PINK   = RGBColor(0xFF, 0x7A, 0xB8)
AMBER  = RGBColor(0xFF, 0xD1, 0x66)
CYAN   = RGBColor(0x6F, 0xD3, 0xFF)
VIOLET = RGBColor(0xB7, 0x9B, 0xFF)
CORAL  = RGBColor(0xFF, 0x8F, 0x6B)
SLATE  = RGBColor(0x9A, 0xA6, 0xBE)

FONT = "Times New Roman"

WIDGET = "/home/mls10/presentation/widget"
GOAL_PANEL = "/home/mls10/logs/goal_search/goal_search_idx108.png"
OUT_PATH = "/home/mls10/presentation/BAIR_LoRA_Presentation.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ------------------------------------------------------------- utilities --
def set_letter_spacing(run, pt_hundredths):
    rPr = run._r.get_or_add_rPr()
    rPr.set("spc", str(pt_hundredths))


def grad(fill, c1, c2, angle=120.0):
    fill.gradient()
    stops = fill.gradient_stops
    stops[0].color.rgb = c1
    stops[0].position = 0.0
    stops[1].color.rgb = c2
    stops[1].position = 1.0
    fill.gradient_angle = angle


def new_slide(prs, c1, c2, angle=125.0):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    grad(slide.background.fill, c1, c2, angle)
    return slide


def add_text(slide, left, top, width, height, text, size, color,
             bold=False, italic=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, line_spacing=1.08, letter_spacing=None,
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
        run.font.name = FONT
        run.font.color.rgb = color
        if letter_spacing is not None:
            set_letter_spacing(run, letter_spacing)
    return tb


def eyebrow(slide, left, top, text, color, align=PP_ALIGN.LEFT, width=Inches(12)):
    add_text(slide, left, top, width, Inches(0.55), text, 30, color,
             bold=True, align=align, letter_spacing=120, upper=True)


def rect(slide, left, top, width, height, fill_color=None, line_color=None,
         radius=False, line_w=1.5, grad_pair=None, grad_angle=90.0):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, left, top, width, height)
    if grad_pair is not None:
        grad(shp.fill, grad_pair[0], grad_pair[1], grad_angle)
    elif fill_color is not None:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill_color
    else:
        shp.fill.background()
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


def accent_bar(slide, left, top, width, c1, c2, height=Pt(4)):
    bar = rect(slide, left, top, width, height, grad_pair=(c1, c2), grad_angle=0.0)
    return bar


def box(slide, left, top, width, height, lines, border, size=30,
        fill=PANEL, text_color=INK):
    rect(slide, left, top, width, height, fill_color=fill, line_color=border,
         radius=True, line_w=2.25)
    add_text(slide, left + Inches(0.08), top, width - Inches(0.16), height,
             lines, size, text_color, bold=True, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.05)


def block_arrow_right(slide, left, top, width, height, color):
    shp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def arrow_conn(slide, x1, y1, x2, y2, color, w=3.0):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    conn.line.color.rgb = color
    conn.line.width = Pt(w)
    ln = conn.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "lg", "len": "lg"})
    ln.append(tail)
    return conn


def chip(slide, left, top, width, height, text, border, size=30):
    rect(slide, left, top, width, height, fill_color=PANEL, line_color=border,
         radius=True, line_w=2.0)
    add_text(slide, left + Inches(0.1), top, width - Inches(0.2), height,
             text, size, INK, bold=True, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)


# ------------------------------------------------------------------ build --
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

# ============================================================ 1 · Title ====
s = new_slide(prs, RGBColor(0x2B, 0x1A, 0x5E), RGBColor(0x0B, 0x3B, 0x43), 130)
add_text(s, Inches(1), Inches(1.75), Inches(11.33), Inches(0.6),
         "BAIR · WAN2.1-1.3B · LORA", 30, CYAN, bold=True,
         align=PP_ALIGN.CENTER, letter_spacing=140)
add_text(s, Inches(1), Inches(2.45), Inches(11.33), Inches(2.0),
         "Action-Conditioned\nWorld Models", 54, INK, bold=True,
         align=PP_ALIGN.CENTER, line_spacing=1.02)
accent_bar(s, Inches(5.27), Inches(4.55), Inches(2.8), PINK, TEAL, Pt(5))
add_text(s, Inches(1.4), Inches(4.95), Inches(10.53), Inches(1.2),
         "Teaching a pretrained video model to obey a command,\nnot just guess the future.",
         30, DIM, align=PP_ALIGN.CENTER, line_spacing=1.2)
add_text(s, Inches(1), Inches(6.35), Inches(11.33), Inches(0.6),
         "Dawidzard  &  Mihajlo", 30, AMBER, bold=True, align=PP_ALIGN.CENTER)

# ======================================================= 2 · Motivation ====
s = new_slide(prs, RGBColor(0x1C, 0x2A, 0x6B), RGBColor(0x0E, 0x4A, 0x4A), 120)
eyebrow(s, Inches(0.8), Inches(0.5), "Motivation", TEAL)
add_text(s, Inches(0.8), Inches(1.15), Inches(11.7), Inches(1.7),
         "A robot that can imagine what happens next\ncan plan before it acts.",
         38, INK, bold=True, line_spacing=1.1)
accent_bar(s, Inches(0.8), Inches(2.95), Inches(3.2), TEAL, CYAN)

fy = Inches(3.35)
box(s, Inches(0.8), fy + Inches(0.75), Inches(2.5), Inches(1.15), "frame", CYAN, 32)
block_arrow_right(s, Inches(3.5), fy + Inches(1.08), Inches(0.85), Inches(0.5), CYAN)
box(s, Inches(4.55), fy + Inches(0.75), Inches(2.5), Inches(1.15), "model", VIOLET, 32)
block_arrow_right(s, Inches(7.25), fy + Inches(1.08), Inches(0.85), Inches(0.5), PINK)
chip(s, Inches(8.3), fy, Inches(3.3), Inches(0.8), "←  left", PINK, 30)
chip(s, Inches(8.3), fy + Inches(0.93), Inches(3.3), Inches(0.8), "↑  up", AMBER, 30)
chip(s, Inches(8.3), fy + Inches(1.86), Inches(3.3), Inches(0.8), "→  right", TEAL, 30)

add_text(s, Inches(0.8), Inches(6.35), Inches(1.7), Inches(0.6), "Goals", 30,
         SLATE, bold=True, upper=True, letter_spacing=80)
gw, gg = Inches(3.25), Inches(0.2)
for i, (g, c) in enumerate([("action control", PINK),
                            ("precision", AMBER),
                            ("failure modes", TEAL)]):
    chip(s, Inches(2.45) + i * (gw + gg), Inches(6.3), gw, Inches(0.75), g, c, 30)

# ============================================ 3 · Architecture diagram =====
s = new_slide(prs, RGBColor(0x12, 0x3A, 0x5C), RGBColor(0x3A, 0x1D, 0x5E), 115)

lane_h = Inches(1.3)
# video lane (top, cyan/teal)
y1 = Inches(1.2)
box(s, Inches(0.6), y1, Inches(2.5), lane_h, "BAIR video\n64 × 64", CYAN)
block_arrow_right(s, Inches(3.3), y1 + Inches(0.4), Inches(0.85), Inches(0.5), CYAN)
box(s, Inches(4.35), y1, Inches(2.1), lane_h, "VAE\nencode", CYAN)
block_arrow_right(s, Inches(6.65), y1 + Inches(0.4), Inches(0.85), Inches(0.5), CYAN)
box(s, Inches(7.7), y1, Inches(2.1), lane_h, "4 latent\nframes", TEAL)

# action lane (bottom, pink/violet)
y2 = Inches(5.35)
box(s, Inches(0.6), y2, Inches(2.5), lane_h, "actions\n4 × 4D", PINK)
block_arrow_right(s, Inches(3.3), y2 + Inches(0.4), Inches(0.85), Inches(0.5), PINK)
box(s, Inches(4.35), y2, Inches(2.1), lane_h, "MLP\n16 → 1536", PINK)
block_arrow_right(s, Inches(6.65), y2 + Inches(0.4), Inches(0.85), Inches(0.5), PINK)
box(s, Inches(7.7), y2, Inches(2.1), lane_h, "+ timestep\nAdaLN", VIOLET)

# the DiT core (center right, violet gradient)
dit = rect(s, Inches(10.15), Inches(2.75), Inches(2.55), Inches(2.0),
           grad_pair=(RGBColor(0x6C, 0x3F, 0xB8), RGBColor(0x2E, 0x7E, 0x8C)),
           line_color=AMBER, radius=True, line_w=2.5, grad_angle=35.0)
add_text(s, Inches(10.15), Inches(2.75), Inches(2.55), Inches(2.0),
         "30 DiT\nblocks\n+ LoRA r16", 30, INK, bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.02)

# converging arrows into the DiT, and out to the decoded video
arrow_conn(s, Inches(9.8), y1 + Inches(0.65), Inches(10.35), Inches(3.05), TEAL)
arrow_conn(s, Inches(9.8), y2 + Inches(0.65), Inches(10.35), Inches(4.45), VIOLET)
box(s, Inches(10.6), Inches(0.55), Inches(2.1), Inches(1.1), "next 4\nlatents", AMBER)
arrow_conn(s, Inches(11.65), Inches(2.75), Inches(11.65), Inches(1.7), AMBER)

add_text(s, Inches(0.6), Inches(3.35), Inches(8.5), Inches(0.7),
         "1.4 % trainable  ·  15 lines changed upstream", 30, AMBER, bold=True)

# ====================================== 4 · The two design decisions =======
s = new_slide(prs, RGBColor(0x3D, 0x17, 0x48), RGBColor(0x14, 0x35, 0x5C), 120)
eyebrow(s, Inches(0.8), Inches(0.5), "Method", VIOLET)
add_text(s, Inches(0.8), Inches(1.15), Inches(11.7), Inches(0.9),
         "Two design decisions.", 40, INK, bold=True)
accent_bar(s, Inches(0.8), Inches(2.1), Inches(3.2), PINK, VIOLET)

cw, ch2 = Inches(5.75), Inches(3.7)
cy = Inches(2.6)
rect(s, Inches(0.8), cy, cw, ch2, fill_color=PANEL, line_color=PINK,
     radius=True, line_w=2.5)
add_text(s, Inches(1.15), cy + Inches(0.4), cw - Inches(0.7), Inches(1.6),
         "Flatten actions,\nnever average", 36, PINK, bold=True, line_spacing=1.08)
add_text(s, Inches(1.15), cy + Inches(2.15), cw - Inches(0.7), Inches(1.3),
         "the mean of “left, then\nback” is zero", 30, DIM, line_spacing=1.15)

rect(s, Inches(6.85), cy, cw, ch2, fill_color=PANEL, line_color=TEAL,
     radius=True, line_w=2.5)
add_text(s, Inches(7.2), cy + Inches(0.4), cw - Inches(0.7), Inches(1.6),
         "Modulate every\nblock, every frame", 36, TEAL, bold=True, line_spacing=1.08)
add_text(s, Inches(7.2), cy + Inches(2.15), cw - Inches(0.7), Inches(1.3),
         "action + timestep → AdaLN,\nzero-init start", 30, DIM, line_spacing=1.15)

# ================================================= 5 · Result: control ====
s = new_slide(prs, RGBColor(0x0F, 0x4D, 0x43), RGBColor(0x1A, 0x1F, 0x5C), 125)
eyebrow(s, Inches(0), Inches(0.6), "Result", TEAL, align=PP_ALIGN.CENTER,
        width=Inches(13.333))
add_text(s, Inches(0), Inches(1.05), Inches(13.333), Inches(2.5),
         "99.6%", 150, INK, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
add_text(s, Inches(1.2), Inches(3.75), Inches(10.93), Inches(0.7),
         "of held-out scenes follow the command  (255 / 256)",
         32, DIM, align=PP_ALIGN.CENTER)

cw, ch, cgap = Inches(2.55), Inches(0.8), Inches(0.25)
row_w = cw * 4 + cgap * 3
cx = (SLIDE_W - row_w) / 2
for glyph, c in [("↑ up ✓", CYAN), ("↓ down ✓", VIOLET),
                 ("← left ✓", PINK), ("→ right ✓", TEAL)]:
    chip(s, cx, Inches(4.75), cw, ch, glyph, c, 30)
    cx += cw + cgap

add_text(s, Inches(1.2), Inches(5.85), Inches(10.93), Inches(0.6),
         "absolute position accuracy: 84.8%", 30, AMBER, bold=True,
         align=PP_ALIGN.CENTER)
add_text(s, Inches(1.2), Inches(6.55), Inches(10.93), Inches(0.6),
         "PSNR 18.6 dB   ·   SSIM 0.79   ·   FID 11.1", 30, DIM,
         align=PP_ALIGN.CENTER)

# ============================================ 6 · Controllability ladder ====
s = new_slide(prs, RGBColor(0x35, 0x19, 0x4F), RGBColor(0x0C, 0x4A, 0x5E), 118)
add_text(s, Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.9),
         "What is the action worth?", 40, INK, bold=True)
accent_bar(s, Inches(0.8), Inches(1.45), Inches(3.2), AMBER, PINK)

chart_left = Inches(0.8)
chart_width = Inches(10.6)
scale_max = 24.0
ceiling_db = 22.74
rows = [
    ("no fine-tune", 7.12, SLATE),
    ("wrong action", 12.45, CORAL),
    ("no action", 13.27, AMBER),
    ("real action", 18.56, TEAL),
]
top = Inches(1.85)
row_step = Inches(1.22)
bar_h = Inches(0.5)
for label, val, color in rows:
    add_text(s, chart_left, top, Inches(6), Inches(0.55), label, 30, DIM,
             bold=True)
    bar_y = top + Inches(0.6)
    rect(s, chart_left, bar_y, chart_width, bar_h, fill_color=PANEL, radius=True)
    bw = Emu(int(chart_width * (val / scale_max)))
    rect(s, chart_left, bar_y, bw, bar_h, fill_color=color, radius=True)
    add_text(s, chart_left + bw + Inches(0.15), bar_y - Inches(0.06),
             Inches(2.2), Inches(0.6), f"{val:.1f} dB", 30, INK, bold=True,
             anchor=MSO_ANCHOR.MIDDLE)
    top += row_step

ceiling_x = chart_left + Emu(int(chart_width * (ceiling_db / scale_max)))
conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, ceiling_x, Inches(1.85),
                              ceiling_x, top - row_step + Inches(1.1))
conn.line.color.rgb = INK
conn.line.width = Pt(2.5)
conn.line.dash_style = MSO_LINE.DASH
add_text(s, ceiling_x - Inches(3.9), Inches(6.75), Inches(4.0), Inches(0.6),
         "VAE ceiling · 22.7 dB", 30, INK, bold=True, align=PP_ALIGN.RIGHT)

# ============================================ 7 · Interactive widget =======
s = new_slide(prs, RGBColor(0x10, 0x3C, 0x5C), RGBColor(0x2C, 0x1A, 0x52), 122)
eyebrow(s, Inches(0.8), Inches(0.45), "Live demo", CYAN)
add_text(s, Inches(0.8), Inches(1.05), Inches(11.7), Inches(0.8),
         "Click a command.", 40, INK, bold=True)

scr = Inches(2.95)
sx, sy = Inches(0.8), Inches(2.15)
rect(s, sx - Inches(0.1), sy - Inches(0.1), scr + Inches(0.2), scr + Inches(0.2),
     fill_color=PANEL, line_color=CYAN, radius=True, line_w=2.0)
s.shapes.add_picture(os.path.join(WIDGET, "screen_context.png"), sx, sy, scr, scr)
add_text(s, sx, sy + scr + Inches(0.15), scr, Inches(0.55), "start frame",
         30, DIM, bold=True, align=PP_ALIGN.CENTER)

link = rect(s, sx - Inches(0.1), sy + scr + Inches(0.75), scr + Inches(0.2),
            Inches(0.75), fill_color=PANEL, line_color=TEAL, radius=True,
            line_w=2.25)
ltf = link.text_frame
ltf.word_wrap = False
ltf.vertical_anchor = MSO_ANCHOR.MIDDLE
lp = ltf.paragraphs[0]
lp.alignment = PP_ALIGN.CENTER
lr = lp.add_run()
lr.text = "FULL DEMO ↗"
lr.font.size = Pt(30)
lr.font.bold = True
lr.font.name = FONT
lr.font.color.rgb = TEAL
link.click_action.hyperlink.address = "arm_control_panel.html"

tile = Inches(1.55)
tgap = Inches(0.13)
px = Inches(4.85)
py = Inches(2.0)
cols = [px, px + tile + tgap, px + 2 * (tile + tgap)]
rws = [py, py + tile + tgap, py + 2 * (tile + tgap)]
pad = [("up", cols[1], rws[0]), ("left", cols[0], rws[1]),
       ("still", cols[1], rws[1]), ("right", cols[2], rws[1]),
       ("down", cols[1], rws[2])]
for name, x, y in pad:
    s.shapes.add_movie(os.path.join(WIDGET, f"tile_{name}.mp4"), x, y, tile, tile,
                       poster_frame_image=os.path.join(WIDGET, f"tile_{name}.png"),
                       mime_type="video/mp4")

chx, chy, chs = Inches(10.35), Inches(2.35), Inches(2.1)
s.shapes.add_movie(os.path.join(WIDGET, "tile_chain.mp4"), chx, chy, chs, chs,
                   poster_frame_image=os.path.join(WIDGET, "tile_chain.png"),
                   mime_type="video/mp4")
add_text(s, chx - Inches(0.35), chy + chs + Inches(0.15), chs + Inches(0.7),
         Inches(0.55), "free rollout", 30, DIM, bold=True,
         align=PP_ALIGN.CENTER)

# ============================================= 8 · Inverse dynamics ========
s = new_slide(prs, RGBColor(0x14, 0x42, 0x4F), RGBColor(0x3B, 0x1E, 0x46), 118)
eyebrow(s, Inches(0.8), Inches(0.45), "Inverse dynamics", VIOLET)
add_text(s, Inches(0.8), Inches(1.05), Inches(11.7), Inches(0.8),
         "Choosing an action by imagining its outcome.", 36, INK, bold=True)

panel_w = Inches(10.4)
panel_h = Inches(2.6)
panel_x = Inches(1.47)
panel_y = Inches(2.05)
rect(s, panel_x - Inches(0.07), panel_y - Inches(0.07), panel_w + Inches(0.14),
     panel_h + Inches(0.14), fill_color=PANEL, line_color=VIOLET, radius=True,
     line_w=2.0)
s.shapes.add_picture(GOAL_PANEL, panel_x, panel_y, panel_w, panel_h)
q = panel_w / 4
for i, (cap, c) in enumerate([("goal", TEAL), ("best", CYAN),
                              ("worst", CORAL), ("score map", AMBER)]):
    add_text(s, panel_x + i * q, panel_y + panel_h + Inches(0.15), q,
             Inches(0.55), cap, 30, c, bold=True, align=PP_ALIGN.CENTER)

rect(s, Inches(0.8), Inches(5.55), Inches(3.4), Inches(1.35), fill_color=PANEL,
     line_color=TEAL, radius=True, line_w=2.25)
add_text(s, Inches(0.8), Inches(5.55), Inches(3.4), Inches(1.35), "15 / 20",
         48, TEAL, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
add_text(s, Inches(4.5), Inches(5.6), Inches(8.0), Inches(1.35),
         "36 imagined futures in one pass, pick the\nclosest to the goal.  Chance: 10 / 20.",
         30, DIM, line_spacing=1.15)

# ===================================================== 9 · Limitations ====
s = new_slide(prs, RGBColor(0x4A, 0x1E, 0x2E), RGBColor(0x1C, 0x20, 0x50), 122)
eyebrow(s, Inches(0.8), Inches(0.5), "Limitations", CORAL)
add_text(s, Inches(0.8), Inches(1.15), Inches(11.7), Inches(0.9),
         "Two failures in free rollout.", 40, INK, bold=True)
accent_bar(s, Inches(0.8), Inches(2.1), Inches(3.2), CORAL, VIOLET)

card_w, card_h = Inches(3.75), Inches(2.6)
cgap = Inches(0.25)
cy = Inches(2.6)
cards = [
    ("98–100%", TEAL, "still obeys"),
    ("79 → 72%", CORAL, "position drifts"),
    ("+97 → +57%", AMBER, "decay halved"),
]
for i, (num, color, label) in enumerate(cards):
    x = Inches(0.8) + i * (card_w + cgap)
    rect(s, x, cy, card_w, card_h, fill_color=PANEL, line_color=color,
         radius=True, line_w=2.5)
    add_text(s, x, cy + Inches(0.45), card_w, Inches(1.0), num, 44, color,
             bold=True, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.2), cy + Inches(1.7), card_w - Inches(0.4),
             Inches(0.7), label, 30, DIM, bold=True, align=PP_ALIGN.CENTER)

add_text(s, Inches(0.8), Inches(5.75), Inches(11.7), Inches(0.9),
         "Exposure bias breaks the image, not obedience.",
         32, INK, bold=True)

# =========================================== 10 · Contributions & next ====
s = new_slide(prs, RGBColor(0x1F, 0x1C, 0x56), RGBColor(0x0F, 0x46, 0x40), 125)
eyebrow(s, Inches(0.8), Inches(0.5), "What we measured", AMBER)
add_text(s, Inches(0.8), Inches(1.15), Inches(11.7), Inches(0.9),
         "Four findings one number would hide.", 38, INK, bold=True)
accent_bar(s, Inches(0.8), Inches(2.1), Inches(3.2), AMBER, TEAL)

items = [
    ("Control and fidelity mature on different clocks", CYAN),
    ("Rollout failure is two separable failures", PINK),
    ("Distillation isn't needed: 6× speed is free", TEAL),
    ("The VAE sets the ceiling: 22.7 dB", AMBER),
]
top = Inches(2.55)
for i, (title, c) in enumerate(items):
    add_text(s, Inches(0.8), top, Inches(0.7), Inches(0.6), f"{i+1}", 34, c,
             bold=True)
    add_text(s, Inches(1.65), top, Inches(11.0), Inches(0.6), title, 30, INK,
             bold=True)
    top += Inches(0.82)

add_text(s, Inches(0.8), Inches(6.15), Inches(11.7), Inches(0.6),
         "Two are negative.  All four we could have kept quiet.",
         30, DIM, italic=True)

prs.save(OUT_PATH)
print("saved:", OUT_PATH, f"{os.path.getsize(OUT_PATH)/1e6:.2f} MB")
