#!/usr/bin/env python3
"""Render a real CLI session as a terminal-styled PNG screenshot.

Run:  python3 scripts/make_screenshot.py
"""

import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

CMD = [sys.executable, "-m", "talentmatch.cli", "match", "frontend_react", "frontend_react_jd"]
proc = subprocess.run(CMD, cwd=ROOT, capture_output=True, text=True)
body = proc.stdout.strip()
lines = [f"$ python -m talentmatch.cli match frontend_react frontend_react_jd"] + body.splitlines()

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 15)

# Measure.
tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
widths = [tmp.textlength(l, font=FONT) for l in lines]
line_h = 22
W = int(max(widths)) + 48
H = line_h * len(lines) + 88

BG, FG, GREEN, TITLEBAR = (13, 17, 23), (201, 209, 217), (63, 185, 80), (22, 27, 34)
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# Title bar with traffic lights.
d.rectangle([0, 0, W, 44], fill=TITLEBAR)
for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
    d.ellipse([18 + i * 26, 15, 32 + i * 26, 29], fill=c)
d.text((W // 2 - 110, 12), "talentmatch — match report", font=BOLD, fill=(139, 148, 158))

y = 60
for i, line in enumerate(lines):
    font = BOLD if i == 0 else FONT
    fill = GREEN if i == 0 else FG
    d.text((24, y), line, font=font, fill=fill)
    y += line_h

out = os.path.join(ASSETS, "cli_screenshot.png")
img.save(out)
print(f"Wrote {out} ({W}x{H})")
