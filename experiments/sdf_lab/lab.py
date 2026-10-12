"""Minimal SDF lab for the 'create from nothing' experiment.

Usage: python lab.py scene.py out.png
A scene file defines `sdf(p)` (p: (N,3) array, y up, +z toward viewer, x = model's left)
and optionally `PARTS = {name: fn}` for per-part text measurements.
Renders a 2x2 orthographic grid (front, side, 3/4, top) and prints a text report.
"""
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

# ---------- the borrowed vocabulary (Shadertoy / Inigo Quilez idioms, numpy edition) ----------

def length(v):
    return np.sqrt((v * v).sum(-1))

def sphere(p, c, r):
    return length(p - np.asarray(c)) - r

def ellipsoid(p, c, r):
    q = (p - np.asarray(c)) / np.asarray(r)
    k0 = length(q)
    k1 = length(q / np.asarray(r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

def box(p, c, half, round=0.0):
    q = np.abs(p - np.asarray(c)) - (np.asarray(half) - round)
    return length(np.maximum(q, 0)) + np.minimum(q.max(-1), 0) - round

def capsule(p, a, b, ra, rb=None):
    """Round cone from a (radius ra) to b (radius rb)."""
    rb = ra if rb is None else rb
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    r = ra + (rb - ra) * h
    return length(pa - h[:, None] * ba) - r

def torus(p, c, R, r):
    q = p - np.asarray(c)
    return length(np.stack([length(q[:, [0, 2]]) - R, q[:, 1]], -1)) - r

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)

def smax(a, b, k):
    return -smin(-a, -b, k)

def cut(a, b, k=1e-4):
    return smax(a, -b, k)

def mirror_x(p):
    q = p.copy()
    q[:, 0] = np.abs(q[:, 0])
    return q

def rot(p, axis, deg, center=(0, 0, 0)):
    """Rotate points by -deg about axis through center (i.e. rotates the shape by +deg)."""
    c = np.asarray(center, float)
    t = np.radians(-deg)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    q = p - c
    out = q.copy()
    out[:, i] = q[:, i] * np.cos(t) - q[:, j] * np.sin(t)
    out[:, j] = q[:, i] * np.sin(t) + q[:, j] * np.cos(t)
    return out + c

def noise3(p, freq=1.0, seed=0):
    """Smooth value noise in [-1,1] (seeded)."""
    q = p * freq
    i = np.floor(q).astype(np.int64)
    f = q - i
    f = f * f * (3 - 2 * f)
    def h(ix, iy, iz):
        n = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
        n = (n ^ (n >> 13)) * 1274126177
        return ((n & 0xFFFF) / 32767.5) - 1.0
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * h(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)
    return out

def _hash3(c, seed):
    out = []
    for k in range(3):
        n = (c[:, 0] * 73856093) ^ (c[:, 1] * 19349663) ^ (c[:, 2] * 83492791) ^ ((seed * 3 + k) * 2654435761)
        n = (n ^ (n >> 13)) * 1274126177
        out.append((n & 0xFFFF) / 65536.0)
    return np.stack(out, -1)

def voronoi(p, cell, seed=0):
    """(F1, F2) distances to the nearest jittered cell points. F2-F1 ~ 0 on cell borders."""
    i = np.floor(p / cell).astype(np.int64)
    f1 = np.full(len(p), 1e9)
    f2 = np.full(len(p), 1e9)
    for o in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T:
        c = i + o
        dist = length((c + 0.15 + 0.7 * _hash3(c, seed)) * cell - p)
        f2 = np.where(dist < f1, f1, np.minimum(f2, dist))
        f1 = np.minimum(f1, dist)
    return f1, f2

def near_surface(d, p, fn, band=0.03):
    """Apply a detail function fn(p_subset) -> offset only near the surface (keeps detail cheap)."""
    m = np.abs(d) < band
    d = d.copy()
    d[m] += fn(p[m])
    return d

# ---------- renderer ----------

BOUND = 1.0  # scene must fit in [-1,1]^3
VIEWS = {  # name: (camera forward direction, up)
    "front": ((0, 0, -1), (0, 1, 0)),
    "side (from +x)": ((-1, 0, 0), (0, 1, 0)),
    "3/4": ((-0.7071, 0, -0.7071), (0, 1, 0)),
    "top": ((0, -1, 0), (0, 0, -1)),
    "back": ((0, 0, 1), (0, 1, 0)),
    "side (from -x)": ((1, 0, 0), (0, 1, 0)),
    "bottom": ((0, 1, 0), (0, 0, 1)),
}
DEFAULT_VIEWS = ["front", "side (from +x)", "3/4", "top"]  # a scene may set VIEW_NAMES
LIGHT = np.array([0.4, 0.7, 0.6]) / np.linalg.norm([0.4, 0.7, 0.6])

def load(path):
    spec = importlib.util.spec_from_file_location("scene", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lab"] = sys.modules[__name__]
    spec.loader.exec_module(mod)
    return mod

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
    origin = np.asarray(center) + (u.ravel()[:, None] * right + v.ravel()[:, None] * up) - fwd * 2 * BOUND
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
        alive[idx[h | (t[idx] > 4 * BOUND)]] = False
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

def render(path, out, res=320, center=(0, 0, 0), half=BOUND):
    names = getattr(load(path), "VIEW_NAMES", DEFAULT_VIEWS)
    cols = 2 if len(names) <= 4 else 3
    with ProcessPoolExecutor(len(names)) as ex:
        tiles = dict(ex.map(render_view, [(path, n, res, center, half) for n in names]))
    grid = Image.new("L", (cols * res, -(-len(names) // cols) * res), 255)
    draw = ImageDraw.Draw(grid)
    for k, name in enumerate(names):
        x, y = (k % cols) * res, (k // cols) * res
        grid.paste(Image.fromarray(tiles[name]), (x, y))
        draw.text((x + 6, y + 4), name, fill=0)
        draw.rectangle([x, y, x + res - 1, y + res - 1], outline=160)
    grid.save(out)

# ---------- mesh export ----------

def _eval_slabs(args):
    path, i0, i1, res, lo, hi = args
    f = load(path).sdf
    s = np.linspace(lo, hi, res)
    g = np.stack(np.meshgrid(s[i0:i1], s, s, indexing="ij"), -1).reshape(-1, 3)
    return f(g).reshape(i1 - i0, res, res)

def export_mesh(path, out, res=320, lo=-BOUND, hi=BOUND):
    from skimage.measure import marching_cubes
    import trimesh
    step = 8
    with ProcessPoolExecutor(16) as ex:
        vol = np.concatenate(list(ex.map(_eval_slabs, [(path, i, min(i + step, res), res, lo, hi) for i in range(0, res, step)])))
    h = (hi - lo) / (res - 1)
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(h, h, h))
    mesh = trimesh.Trimesh(verts + lo, faces)
    if mesh.volume < 0:
        mesh.invert()
    mesh.export(out)
    return f"mesh {out}: v={len(mesh.vertices)} f={len(mesh.faces)} watertight={mesh.is_watertight} bodies={mesh.body_count}"

# ---------- text senses ----------

def fmt(v):
    return ",".join(f"{x:+.2f}" for x in v)

def report(mod, n=96):
    s = np.linspace(-BOUND, BOUND, n)
    g = np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3)
    lines = []
    for name, f in [("ALL", mod.sdf)] + list(getattr(mod, "PARTS", {}).items()):
        inside = g[f(g) < 0]
        if not len(inside):
            lines.append(f"{name}: empty")
            continue
        lo, hi = inside.min(0), inside.max(0)
        vol = len(inside) * (2 * BOUND / (n - 1)) ** 3
        lines.append(f"{name}: min={fmt(lo)} max={fmt(hi)} size={fmt(hi - lo)} vol={vol:.3f}")
    d = mod.sdf(g)
    from scipy import ndimage
    lab_, k = ndimage.label((d < 0).reshape(n, n, n))
    if k > 1:  # report every piece not connected to the largest one, with the part it belongs to
        sizes = ndimage.sum(np.ones_like(lab_), lab_, range(1, k + 1))
        main = 1 + int(np.argmax(sizes))
        for c in range(1, k + 1):
            if c == main:
                continue
            pts = g[(lab_ == c).ravel()]
            if len(pts) < 4:  # thinner than the grid (spike tips): sampling noise
                continue
            owner = min(getattr(mod, "PARTS", {}).items(), key=lambda kv: kv[1](pts).mean(), default=("?", None))[0]
            lines.append(f"FLOATING piece ({owner}) at {fmt(pts.mean(0))} vox={len(pts)}")
    lines.append(f"components={k}")
    shell = np.abs(d) < 0.03
    m = g[shell].copy()
    m[:, 0] *= -1
    lines.append(f"symmetry_err(x)={np.abs(mod.sdf(m) - d[shell]).mean():.4f}  clipped_at_bound={bool((g[d < 0].__abs__() > BOUND - 0.03).any())}")
    return "\n".join(lines)

if __name__ == "__main__":
    scene, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    if out.endswith(".glb"):
        print(export_mesh(scene, out, int(sys.argv[3]) if len(sys.argv) > 3 else 320), f"{time.time() - t0:.1f}s")
        sys.exit()
    if len(sys.argv) > 3:  # zoom: cx,cy,cz,half
        *c, h = map(float, sys.argv[3].split(","))
        render(scene, out, center=c, half=h)
    else:
        render(scene, out)
    print(report(load(scene)))
    print(f"render+report {time.time() - t0:.1f}s, scene {len(open(scene, encoding='utf-8').read())} chars (~{len(open(scene, encoding='utf-8').read()) // 4} tokens)")
