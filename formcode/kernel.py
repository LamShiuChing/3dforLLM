"""The modeling vocabulary: SDF primitives and blends, curves, landmark fitting, sculpt strokes, anatomy parts.

Units are meters. Axes: y up, +z toward the viewer (the subject's front), x = the subject's left.
Names ending in _L are mirrored to _R. A scene is a Python file that does `from formcode import *` and defines `sdf(p)`.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

_POOL = None


def peval(f, p, n=32768):
    """f(p) in chunks on a thread pool (numpy releases the GIL): the same values as f(p), every point is independent.
    Inside a pool thread it evaluates directly (no nested waits)."""
    # ponytail: an sdf is ~450 numpy passes per call; the GIL and that overhead cap threads at ~3-5x (numba would fuse them)
    global _POOL
    if len(p) <= n or threading.current_thread().name.startswith("peval"):
        return f(p)
    if _POOL is None:
        _POOL = ThreadPoolExecutor(min(16, os.cpu_count() or 1), thread_name_prefix="peval")
    return np.concatenate(list(_POOL.map(f, [p[j:j + n] for j in range(0, len(p), n)])))

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
    DETAIL.add("near_surface")
    m = np.abs(d) < band
    d = d.copy()
    d[m] += fn(p[m])
    return d

# ---------- geometry helpers ----------

def with_mirrors(L):
    out = {k: np.asarray(v, float) for k, v in L.items()}
    for k, v in list(out.items()):
        if k.endswith("_L"):
            out[k[:-2] + "_R"] = v * (-1, 1, 1)
    return out


def spline(pts, step):
    """Catmull-Rom through pts, resampled about every `step` meters."""
    P = np.asarray(pts, float)
    if len(P) == 1:
        return P
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1:i + 3]
        n = max(2, int(np.ceil(np.linalg.norm(p2 - p1) / step)))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2])
    return np.array(out)


def wendland(r):
    r = np.clip(r, 0, 1)
    return (1 - r) ** 4 * (4 * r + 1)


def dist_to_curve(p, C):
    d = np.full(len(p), 1e9)
    for a, b in zip(C[:-1], C[1:]):
        ba = b - a
        h = np.clip(((p - a) @ ba) / (ba @ ba), 0, 1)
        d = np.minimum(d, length(p - a - h[:, None] * ba))
    return d


def on_surface(f, pts, iters=4):
    """Snap points onto the zero surface of f by gradient steps."""
    q = np.asarray(pts, float).copy()
    for _ in range(iters):
        e = 1e-4
        g = np.stack([f(q + d) - f(q - d) for d in np.eye(3) * e], -1) / (2 * e)
        q -= f(q)[:, None] * g / np.maximum((g * g).sum(-1, keepdims=True), 1e-9)
    return q


def cast(f, origin, direction, t_max=0.6, eps=1e-5):
    """March rays from origin(s) along direction; returns the first surface hit points."""
    o = np.atleast_2d(np.asarray(origin, float))
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    t = np.zeros(len(o))
    for _ in range(400):
        step = f(o + t[:, None] * d)
        t += np.maximum(step * 0.5, eps)
        if (np.abs(step) < eps).all() or (t > t_max).all():
            break
    return o + t[:, None] * d


ATTACHED = []  # (name, part_sdf, host_sdf) registered by eye()/ear(): checked by the seating sense
TYPICAL = {    # anatomy-scale lint: name -> (measure, lo, hi) in meters (adult)
    "nostril": ("length", 0.009, 0.014),
    "nasolabial": ("width", 0.006, 0.012),
    "philtrum": ("width", 0.008, 0.012),
    "mouth_slit": ("length", 0.045, 0.058),
    "upper_lid_crease": ("width", 0.003, 0.006),
    "eye_fissure": ("length", 0.027, 0.033),
    "ear": ("length", 0.055, 0.068),
}
LINT = []
EDGES = []      # (points (M,3), radius): where sharp edges are intended (grooves, carves, eyes, ears); the kink sense skips them
DETAIL = set()  # surface-detail functions in use (near_surface): kinks are expected everywhere
TRIMS = []      # front_outline polygons (M,2): their corners crease the surface, so kink warnings name them as the cause


def _edge(points, radius, sym):
    P = np.atleast_2d(np.asarray(points, float))
    EDGES.append((np.vstack([P, P * (-1, 1, 1)]) if sym else P, float(radius)))


def _lint(name, measure, value):
    if name in TYPICAL and TYPICAL[name][0] == measure:
        _, lo, hi = TYPICAL[name]
        if not lo <= value <= hi:
            LINT.append(f"{name} {measure} {value * 1000:.1f}mm (typical {lo * 1000:.0f}-{hi * 1000:.0f})")


_TYPICAL0 = dict(TYPICAL)


def reset():
    """Clear what the last scene registered (called before each scene load; a scene may edit TYPICAL)."""
    ATTACHED.clear()
    LINT.clear()
    EDGES.clear()
    DETAIL.clear()
    TRIMS.clear()
    TYPICAL.clear()
    TYPICAL.update(_TYPICAL0)


# ---------- vocabulary ----------

def fit(base, targets, radius=0.022, lam=1e-7):
    """Bend `base` so its surface passes through every target. Returns (sdf, per-target correction in m).
    radius: scalar or one per target (bigger = smoother, wider reshape around isolated landmarks)."""
    T = np.asarray(targets, float)
    radius = np.broadcast_to(np.asarray(radius, float), (len(T),))
    Phi = wendland(np.linalg.norm(T[:, None] - T[None], axis=-1) / radius[None])
    d0 = base(T)
    w = np.linalg.solve(Phi + lam * np.eye(len(T)), d0)
    tree, rmax = cKDTree(T), radius.max()

    def apply(v, p):
        if not len(p):
            return v
        pr = cKDTree(p).sparse_distance_matrix(tree, rmax, output_type="ndarray")  # wendland is 0 past the radius:
        o = np.lexsort((pr["j"], pr["i"]))  # only near (point, target) pairs add, each point in one order (exact in chunks)
        i, j = pr["i"][o], pr["j"][o]
        r = np.linalg.norm(p[i] - T[j], axis=-1) / radius[j]
        return v - np.bincount(i, wendland(r) * w[j], minlength=len(p))
    f = lambda p: apply(base(p), p)
    f.stage = ("fit", (T.tobytes(), radius.tobytes(), d0.tobytes()), apply, None)
    return f, d0


def sculpt(f, ops):
    """ops: (name, kind, points, amount, width_or_radius, sym). kind in crease|ridge|inflate|carve|groove.
    groove: true cut along the curve (a slit), amount = tube radius, width = blend k.
    crease/ridge: along the curve through points (snapped to surface), depth/height = amount, falloff = width.
    inflate: at points[0], radius = width, amount = outward push.  carve: ellipsoid cut, amount = radii, width = blend k.
    sym=True applies the op on both sides (define it on the left, x>0)."""
    built, names = [], []
    snap = [np.asarray(pts, float) for _, kind, pts, *_ in ops if kind != "carve"]
    snap = iter(np.split(on_surface(f, np.concatenate(snap)), np.cumsum([len(s) for s in snap])[:-1]) if snap else [])
    for name, kind, pts, amount, width, sym in ops:  # (all points snapped in one call: same points, fewer calls)
        names.append(name)
        if kind == "carve":
            built.append((kind, np.asarray(pts[0], float), amount, width, sym))
            _lint(name, "length", 2 * max(amount))
            _lint(name, "width", 2 * min(amount))
            _edge(pts[0], 1.5 * (max(amount) + width), sym)
            continue
        S = next(snap)
        C = spline(S, width / 2) if kind in ("crease", "ridge", "groove") else S
        _lint(name, "width", width)
        _lint(name, "length", float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()))
        reach = width + (amount if kind == "groove" else 0)
        if kind == "groove":
            _edge(C, max(2 * reach, 0.002), sym)
        lo, hi = C.min(0) - reach, C.max(0) + reach
        built.append((kind, C, amount, width, sym, lo, hi))

    def apply(v, p):
        d = v.copy()
        for op in built:
            kind, C, amount, width, sym = op[:5]
            q = mirror_x(p) if sym else p
            if kind == "carve":
                d = cut(d, ellipsoid(q, C, amount), width)
                continue
            lo, hi = op[5:]
            m = np.all((q > lo) & (q < hi), -1)  # only touch points near the op
            if not m.any():
                continue
            if kind == "groove":
                d[m] = cut(d[m], dist_to_curve(q[m], C) - amount, width)
                continue
            if kind == "inflate":
                off = -amount * wendland(length(q[m] - C[0]) / width)
            else:
                off = (amount if kind == "crease" else -amount) * wendland(dist_to_curve(q[m], C) / width)
            d[m] += off
        return d
    g = lambda p: apply(f(p), p)
    # per op: a signature and the box it changes (None = anywhere: a carve is an unmasked cut)
    elems = [(n, repr([np.asarray(x).tobytes() if isinstance(x, np.ndarray) else x for x in op]),
              None if op[0] == "carve" else (op[5], op[6], op[4])) for n, op in zip(names, built)]
    g.stage = ("strokes", tuple(e[1] for e in elems), apply, elems)
    return g


def polygon2d(q, V):
    """Signed distance in 2D from points q (N,2) to a closed polygon V (M,2) (Inigo Quilez's sdPolygon)."""
    d = ((q - V[0]) ** 2).sum(-1)
    s = np.ones(len(q))
    for i in range(len(V)):
        a, b = V[i - 1], V[i]
        e, w = a - b, q - b
        h = np.clip((w @ e) / (e @ e), 0, 1)
        d = np.minimum(d, ((w - e * h[:, None]) ** 2).sum(-1))
        c1, c2 = q[:, 1] >= b[1], q[:, 1] < a[1]
        c3 = e[0] * w[:, 1] > e[1] * w[:, 0]
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s[flip] *= -1
    return s * np.sqrt(d)


def front_outline(f, halfwidths, k=0.006):
    """Trim the shape to a front-view silhouette (visual hull). halfwidths: [(y, w), ...] top to bottom, symmetric.
    Above the first and below the last entry nothing is trimmed."""
    hw = np.asarray(halfwidths, float)
    right = [(w, y) for y, w in hw]
    top, bot = hw[0, 0] + 1.0, hw[-1, 0] - 1.0
    V = np.array([(5.0, top)] + right + [(hw[-1, 1], bot), (-hw[-1, 1], bot)] + [(-w, y) for w, y in right[::-1]] + [(-5.0, top)])
    TRIMS.append(V)
    return lambda p: smax(f(p), polygon2d(p[:, :2], V), k)


def _circle3(a, b, c):
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    den = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / den
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / den
    return np.array([ux, uy]), np.hypot(ax - ux, ay - uy)


def eye(f, center, r, en, ex, up, lo, lid=0.0018, recess=0.0005):
    """Left eye (mirrored): eyeball + cornea seated behind an almond opening.
    center = (x, y): depth is NOT given — the cornea apex is placed `recess` behind the skin found at (x, y).
    en/ex = inner/outer canthus, up = upper-lid peak, lo = lower-lid low point (x,y used; the eye looks along +z)."""
    skin_z = cast(f, (center[0], center[1], 0.3), (0, 0, -1))[0, 2]
    c = np.array([center[0], center[1], skin_z - 1.1 * r - recess])
    _edge(c, 2 * r, True)
    _lint("eye_fissure", "length", float(np.hypot(ex[0] - en[0], ex[1] - en[1])))
    cu, ru = _circle3(en[:2], up[:2], ex[:2])
    cl, rl = _circle3(en[:2], lo[:2], ex[:2])

    def apply(v, p):
        q = mirror_x(p)
        ap = np.maximum(length(q[:, :2] - cu) - ru, length(q[:, :2] - cl) - rl)  # almond in x,y
        ap = smax(ap, np.abs(q[:, 2] - (c[2] + r)) - 0.9 * r, 0.001)            # extrude in front of the eye
        d = smin(v, sphere(q, c, r + lid), 0.004)                                 # lids wrap the ball
        d = cut(d, ap, 0.0008)                                                    # open the almond
        ball = np.minimum(sphere(q, c, r), sphere(q, c + (0, 0, 0.42 * r), 0.68 * r))  # eyeball + cornea
        return np.minimum(d, ball)
    g = lambda p: apply(f(p), p)
    g.stage = ("eye", np.array([*c, *cu, ru, *cl, rl, r, lid]).tobytes(), apply, None)
    return g


def ear(f, tragion, flare=0.004, size=1.0, splay=0.0):
    """Left ear (mirrored). tragion = (y, z): x is NOT given — the ear root is snapped onto the head surface.
    flare = how far the rim stands off the head; size scales the ear (1.0 = adult, ~62 mm long);
    splay = degrees the ear swings outward around its root (keeps it attached, unlike flare)."""
    t = cast(f, (0.3, tragion[0], tragion[1]), (-1, 0, 0))[0]
    _lint("ear", "length", 2 * 0.031 * size)
    S = lambda *v: np.array(v) * size

    def part(p):
        q0 = rot(mirror_x(p), "x", 12, center=t)            # root stays fused to the head
        q = rot(rot(mirror_x(p), "y", splay, center=t), "x", 12, center=t)  # shell swings out around the root
        shell = ellipsoid(q, t + (flare, 0, 0) + S(0, 0.010, -0.014), S(0.007, 0.031, 0.019))
        concha = ellipsoid(q, t + (flare, 0, 0) + S(0.006, 0.003, -0.011), S(0.006, 0.015, 0.010))
        scapha = ellipsoid(q, t + (flare, 0, 0) + S(0.006, 0.018, -0.017), S(0.005, 0.014, 0.011))
        lobe = ellipsoid(q, t + (flare, 0, 0) + S(0.001, -0.018, -0.007), S(0.005, 0.010, 0.008))
        root = ellipsoid(q0, t + S(0.0, 0.006, -0.008), S(0.006, 0.024, 0.012))  # front third fused to the head
        return smin(smin(cut(cut(shell, concha, 0.003), scapha, 0.002), lobe, 0.004), root, 0.004)

    part.sig = np.array([*t, flare, size, splay], float).tobytes()  # what the part is (the measure caches)
    ATTACHED.append(("ear", part, f))
    _edge(t, 0.04 * size, True)
    apply = lambda v, p: smin(v, part(p), 0.005)
    g = lambda p: apply(f(p), p)
    g.stage = ("ear", part.sig, apply, None)
    return g


