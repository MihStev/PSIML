#!/usr/bin/env python
"""One-off asset prep for the Q&A slide: square-crop the real headshots and
render scannable QR codes for both LinkedIn URLs. Run once (or whenever the
source photos/links change); outputs are consumed by build_presentation_pptx.py.
"""
import os
from PIL import Image
import qrcode

PRES = "/home/mls10/presentation"

PEOPLE = [
    dict(src="Mihajlo_slika.jpeg", photo_out="qa_mihajlo_sq.jpg",
         qr_out="qa_qr_mihajlo.png", top_bias=0.25,
         url="https://www.linkedin.com/in/mihajlo-stevanovi%C4%87/"),
    dict(src="David_slika.jpeg", photo_out="qa_david_sq.jpg",
         qr_out="qa_qr_david.png", top_bias=0.5,
         url="https://www.linkedin.com/in/david-markovic-3a107a309/"),
]

for p in PEOPLE:
    im = Image.open(os.path.join(PRES, p["src"])).convert("RGB")
    w, h = im.size
    side = min(w, h)
    if w >= h:
        # landscape: center crop horizontally
        left = (w - side) * 0.5
    else:
        # portrait: bias toward the top so heads aren't clipped
        left = 0
    if h >= w:
        top = (h - side) * p["top_bias"]
    else:
        top = (h - side) * 0.5
    im.crop((int(left), int(top), int(left) + side, int(top) + side)).save(
        os.path.join(PRES, p["photo_out"]), quality=92)
    print("photo:", p["photo_out"], side, "x", side)

    qr = qrcode.QRCode(border=1, box_size=12, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(p["url"])
    qr.make(fit=True)
    qr.make_image(fill_color="black", back_color="white").save(os.path.join(PRES, p["qr_out"]))
    print("qr:", p["qr_out"], "->", p["url"])
