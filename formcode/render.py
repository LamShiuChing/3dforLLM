"""Scene loading, sphere-traced multi-view renders, mesh export.

A scene may set: FRAME = (cx, cy, cz, half) (the cube that is rendered, exported and measured; default the unit cube),
VIEW_NAMES, LANDMARKS (drawn as red dots), and the tracer knobs EPS, STEP, MAX_STEPS, AO.
"""
import importlib.util
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from . import kernel
from .kernel import length

FRAME = (0.0, 0.0, 0.0, 1.0)
VIEWS = {  # name: (camera forward direction, up)
    "front": ((0, 0, -1), (0, 1, 0)),
    "side (from +x)": ((-1, 0, 0), (0, 1, 0)),
    "3/4": ((-0.7071, 0, -0.7071), (0, 1, 0)),
    "top": ((0, -1, 0), (0, 0, -1)),
    "back": ((0, 0, 1), (0, 1, 0)),
    "side (from -x)": ((1, 0, 0), (0, 1, 0)),
    "bottom": ((0, 1, 0), (0, 0, 1)),
}
DEFAULT_VIEWS = ["front", "side (from +x)", "3/4", "top"]
SHORT = {"glabella": "g", "nasion": "n", "pronasale": "prn", "subnasale": "sn", "labrale_sup": "ls",
         "stomion": "sto", "labrale_inf": "li", "sublabiale": "sl", "pogonion": "pg", "menton": "me"}


def load(path):
    if str(path).endswith(".fc"):  # an engine command script: build it
        from .engine import load as load_fc
        return load_fc(path)
    kernel.reset()
    spec = importlib.util.spec_from_file_location("scene", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frame(mod):
    *c, h = getattr(mod, "FRAME", FRAME)
    return np.asarray(c, float), float(h)


def normals(f, p, e=1e-3):
    n = np.stack([f(p + d) - f(p - d) for d in np.eye(3) * e], -1)
    return n / np.maximum(length(n)[:, None], 1e-9)


def render_view(args):
    path, name, res, center, half = args
    mod = load(path)
    f = mod.sdf
    eps, step_k, max_steps = getattr(mod, "EPS", 1e-3), getattr(mod, "STEP", 0.9), getattr(mod, "MAX_STEPS", 160)
    fwd, up = (np.asarray(v, float) for v in VIEWS[name])
    right = np.cross(fwd, up)
    s = np.linspace(-half, half, res)
    u, v = np.meshgrid(s, -s)
    origin = np.asarray(center) + (u.ravel()[:, None] * right + v.ravel()[:, None] * up) - fwd * 2
    t = np.zeros(len(origin))
    hit = np.zeros(len(origin), bool)
    alive = np.ones(len(origin), bool)
    for _ in range(max_steps):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        d = f(origin[idx] + t[idx, None] * fwd)
        t[idx] += d * step_k
        h = d < eps
        hit[idx[h]] = True
        alive[idx[h | (t[idx] > 4)]] = False
    img = np.full(len(origin), 235.0)
    if hit.any():
        p = origin[hit] + t[hit, None] * fwd
        n = normals(f, p, e=eps)
        light = -fwd + 0.6 * up + 0.4 * right  # moves with the camera: every view is readable
        diff = np.clip(n @ (light / np.linalg.norm(light)), 0, 1)
        a = getattr(mod, "AO", 0.03)
        ao = np.clip(f(p + n * a) / a, 0, 1)  # cheap cavity darkening
        img[hit] = 255 * (0.15 + 0.65 * diff + 0.2 * ao) * (0.6 + 0.4 * ao)
    return name, np.clip(img, 0, 255).reshape(res, res).astype(np.uint8)


def render(path, out, res=320, center=None, half=None, labels=True):
    """Grid of views. center/half default to the scene's FRAME. Landmarks are drawn when the scene defines LANDMARKS."""
    mod = load(path)
    c, h = frame(mod)
    center = c if center is None else np.asarray(center, float)
    half = h if half is None else half
    names = getattr(mod, "VIEW_NAMES", DEFAULT_VIEWS)
    cols = 2 if len(names) <= 4 else 3
    with ProcessPoolExecutor(len(names)) as ex:
        tiles = dict(ex.map(render_view, [(path, n, res, center, half) for n in names]))
    grid = Image.new("RGB", (cols * res, -(-len(names) // cols) * res), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    L = getattr(mod, "LANDMARKS", {}) if labels else {}
    for k, name in enumerate(names):
        x0, y0 = (k % cols) * res, (k // cols) * res
        grid.paste(Image.fromarray(tiles[name]).convert("RGB"), (x0, y0))
        fwd, up = (np.asarray(v, float) for v in VIEWS[name])
        right = np.cross(fwd, up)
        for lname, v in L.items():
            if name not in ("front", "side (from +x)", "3/4") or (name != "front" and v[0] < -1e-4) or lname.endswith("_R"):
                continue
            rel = v - center
            px = x0 + (rel @ right + half) / (2 * half) * (res - 1)
            py = y0 + (half - rel @ up) / (2 * half) * (res - 1)
            if x0 <= px < x0 + res and y0 <= py < y0 + res:
                draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(220, 40, 40))
                if name == "side (from +x)":
                    draw.text((px + 3, py - 5), SHORT.get(lname, ""), fill=(200, 30, 30))
        draw.text((x0 + 6, y0 + 4), name, fill=(0, 0, 0))
        draw.rectangle([x0, y0, x0 + res - 1, y0 + res - 1], outline=(160, 160, 160))
    grid.save(out)


def compare(a, b, out, labels=("before", "after")):
    """Two images side by side."""
    A, B = Image.open(a), Image.open(b)
    m = Image.new("RGB", (A.width + B.width, max(A.height, B.height)), (255, 255, 255))
    m.paste(A, (0, 0))
    m.paste(B, (A.width, 0))
    d = ImageDraw.Draw(m)
    d.text((6, A.height - 14), labels[0], fill=(0, 0, 255))
    d.text((A.width + 6, B.height - 14), labels[1], fill=(0, 0, 255))
    m.save(out)


def _eval_slabs(args):
    path, i0, i1, res, lo, h = args
    f = load(path).sdf
    s = np.linspace(0, 2 * h, res)
    g = np.stack(np.meshgrid(lo[0] + s[i0:i1], lo[1] + s, lo[2] + s, indexing="ij"), -1).reshape(-1, 3)
    n = 32768  # bounded memory: fit() builds (points x 48 x 3) temporaries
    return np.concatenate([f(g[j:j + n]) for j in range(0, len(g), n)]).reshape(i1 - i0, res, res)


def export_mesh(path, out, res=320):
    """Marching cubes over the scene's FRAME. Needs scikit-image and trimesh (pip install formcode[mesh])."""
    from skimage.measure import marching_cubes
    import trimesh
    c, h = frame(load(path))
    lo = c - h
    step = 8
    with ProcessPoolExecutor() as ex:
        vol = np.concatenate(list(ex.map(_eval_slabs, [(path, i, min(i + step, res), res, lo, h) for i in range(0, res, step)])))
    sp = 2 * h / (res - 1)
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(sp, sp, sp))
    mesh = trimesh.Trimesh(verts + lo, faces)
    if mesh.volume < 0:
        mesh.invert()
    mesh.export(out)
    return f"mesh {out}: v={len(mesh.vertices)} f={len(mesh.faces)} watertight={mesh.is_watertight} bodies={mesh.body_count}"
