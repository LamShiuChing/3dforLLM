"""Overlay a labeled pixel grid on a crop of the reference so the LLM can read landmark coordinates as numbers.
usage: python grid.py reference.png x0 y0 x1 y1 out.png [scale]"""
import sys

from PIL import Image, ImageDraw

src, x0, y0, x1, y1, out = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
k = int(sys.argv[7]) if len(sys.argv) > 7 else 2
im = Image.open(src).convert("RGB").crop((x0, y0, x1, y1)).resize(((x1 - x0) * k, (y1 - y0) * k), Image.LANCZOS)
d = ImageDraw.Draw(im, "RGBA")
for x in range((x0 // 10 + 1) * 10, x1, 10):
    major = x % 50 == 0
    d.line([((x - x0) * k, 0), ((x - x0) * k, im.height)], fill=(0, 120, 255, 110 if major else 40), width=1)
    if major:
        d.text(((x - x0) * k + 2, 2), str(x), fill=(0, 60, 200, 255))
for y in range((y0 // 10 + 1) * 10, y1, 10):
    major = y % 50 == 0
    d.line([(0, (y - y0) * k), (im.width, (y - y0) * k)], fill=(255, 60, 0, 110 if major else 40), width=1)
    if major:
        d.text((2, (y - y0) * k + 2), str(y), fill=(200, 40, 0, 255))
im.save(out)
