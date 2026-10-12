"""Compare a scene against the reference sheet: pixel-aligned renders, reference line art overlaid in red,
and text deviations of the outline per named band (mm). usage: python refcompare.py scene.py out.png"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdf_lab"))
from lab import load, render_view  # noqa: E402

REF = Path(__file__).with_name("reference.png")
S = 0.000564                      # meters per reference pixel (skull top -> chin = 390 px = 0.22 m)
EYE_Y, FRONT_CX, SIDE_CZ = 350, 446, 1080
HALF_PX = 260
CENTER_Y_PX = 360
FRONT_BOX = (FRONT_CX - HALF_PX, CENTER_Y_PX - HALF_PX, FRONT_CX + HALF_PX, CENTER_Y_PX + HALF_PX)
SIDE_BOX = (SIDE_CZ - HALF_PX, CENTER_Y_PX - HALF_PX, SIDE_CZ + HALF_PX, CENTER_Y_PX + HALF_PX)
CENTER = (0.0, (EYE_Y - CENTER_Y_PX) * S, 0.0)
HALF = HALF_PX * S
RES = 2 * HALF_PX

# bands in reference pixel rows (name, y0, y1): where the outline is compared
SIDE_BANDS = [("forehead", 180, 280), ("brow/nasion", 280, 350), ("nose", 350, 415), ("lips", 415, 470),
              ("chin", 470, 505), ("throat", 520, 600)]
FRONT_BANDS = [("forehead skin", 180, 300), ("ears", 345, 420), ("jaw", 430, 470), ("lower jaw", 470, 505)]


def ref_masks():
    im = np.asarray(Image.open(REF).convert("RGB")).astype(int)
    ink = (im[..., 0] > 180) & (im[..., 1] > 100) & (im[..., 2] < im[..., 1] + 5) & (im.min(-1) < 235)  # skin: B<G (hair highlights have B>G)
    lines = im.max(-1) < 110               # dark line art
    crop = lambda a, b: a[b[1]:b[3], b[0]:b[2]]
    return {"front": (crop(ink, FRONT_BOX), crop(lines, FRONT_BOX)), "side": (crop(ink, SIDE_BOX), crop(lines, SIDE_BOX))}


def edges(mask, rows, side):
    """Per row: leftmost/rightmost filled column (or NaN)."""
    out = []
    for r in rows:
        cols = np.nonzero(mask[r])[0]
        out.append((cols.min() if side == "min" else cols.max()) if len(cols) else np.nan)
    return np.array(out, float)


def main(scene, out):
    t0 = time.time()
    with ProcessPoolExecutor(2) as ex:
        tiles = dict(ex.map(render_view, [(scene, v, RES, CENTER, HALF) for v in ("front", "side (from +x)")]))
    ref = ref_masks()
    mine = {"front": tiles["front"] < 230, "side": tiles["side (from +x)"] < 230}
    lines = ["(outlines compare SKIN only; hair is not modeled)"]
    # side: front-most edge (smallest column = most forward) per band
    for name, y0, y1 in SIDE_BANDS:
        rows = np.arange(y0, y1) - SIDE_BOX[1]
        d = (edges(mine["side"], rows, "min") - edges(ref["side"][0], rows, "min")) * S * 1000  # + = mine is behind
        if np.isfinite(d).any():
            lines.append(f"side {name:12s} mean {np.nanmean(d):+5.1f}mm  worst {d[np.nanargmax(np.abs(d))]:+5.1f}mm (+ = mine behind ref)")
    # front: half-width per band (right edge = subject's left), compares outline incl. ears
    for name, y0, y1 in FRONT_BANDS:
        rows = np.arange(y0, y1) - FRONT_BOX[1]
        d = (edges(mine["front"], rows, "max") - edges(ref["front"][0], rows, "max")) * S * 1000  # + = mine wider
        if np.isfinite(d).any():
            lines.append(f"front {name:11s} mean {np.nanmean(d):+5.1f}mm  worst {d[np.nanargmax(np.abs(d))]:+5.1f}mm (+ = mine wider)")
    # picture: my render, reference line art in red; reference to the right for context
    panels = []
    for view, tile in (("front", tiles["front"]), ("side", tiles["side (from +x)"])):
        rgb = np.stack([tile] * 3, -1).astype(np.uint8)
        rgb[ref[view][1]] = (220, 30, 30)
        panels.append(rgb)
    refimg = np.asarray(Image.open(REF).convert("RGB"))
    panels += [refimg[FRONT_BOX[1]:FRONT_BOX[3], FRONT_BOX[0]:FRONT_BOX[2]], refimg[SIDE_BOX[1]:SIDE_BOX[3], SIDE_BOX[0]:SIDE_BOX[2]]]
    top = np.concatenate(panels[:2], 1)
    bottom = np.concatenate(panels[2:], 1)
    Image.fromarray(np.concatenate([top, bottom], 0)).resize((800, 800)).save(out)
    print("\n".join(lines))
    print(f"compare {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
