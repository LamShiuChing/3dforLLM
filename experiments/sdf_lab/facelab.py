"""Face lab: an anatomical vocabulary on top of lab.py (experiment 03).

LLM-facing vocabulary (meters, +z front, y up, x = subject's left; names ending _L are mirrored to _R):
  landmarks + midline profile -> fit(): the surface is bent to pass exactly through them (RBF correction)
  sculpt ops along curves between points: crease / ridge / inflate / carve   (curves are snapped onto the surface)
  eye(): eyeball seated behind an almond opening defined by canthi + lid points
Senses (text first): fit corrections, landmark drift, profile deviation; labeled renders; side-by-side compare.

Usage: python facelab.py face.py out.png [cx,cy,cz,half]
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from lab import *  # noqa: F401,F403
from lab import DEFAULT_VIEWS, VIEWS, load, render_view

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


def _lint(name, measure, value):
    if name in TYPICAL and TYPICAL[name][0] == measure:
        _, lo, hi = TYPICAL[name]
        if not lo <= value <= hi:
            LINT.append(f"{name} {measure} {value * 1000:.1f}mm (typical {lo * 1000:.0f}-{hi * 1000:.0f})")


# ---------- vocabulary ----------

def fit(base, targets, radius=0.022, lam=1e-7):
    """Bend `base` so its surface passes through every target. Returns (sdf, per-target correction in m).
    radius: scalar or one per target (bigger = smoother, wider reshape around isolated landmarks)."""
    T = np.asarray(targets, float)
    radius = np.broadcast_to(np.asarray(radius, float), (len(T),))
    Phi = wendland(np.linalg.norm(T[:, None] - T[None], axis=-1) / radius[None])
    d0 = base(T)
    w = np.linalg.solve(Phi + lam * np.eye(len(T)), d0)

    def f(p):
        acc = np.zeros(len(p))
        for j in range(0, len(T), 48):
            r = np.linalg.norm(p[:, None, :] - T[None, j:j + 48], axis=-1) / radius[None, j:j + 48]
            acc += wendland(r) @ w[j:j + 48]
        return base(p) - acc
    return f, d0


def sculpt(f, ops):
    """ops: (name, kind, points, amount, width_or_radius, sym). kind in crease|ridge|inflate|carve|groove.
    groove: true cut along the curve (a slit), amount = tube radius, width = blend k.
    crease/ridge: along the curve through points (snapped to surface), depth/height = amount, falloff = width.
    inflate: at points[0], radius = width, amount = outward push.  carve: ellipsoid cut, amount = radii, width = blend k.
    sym=True applies the op on both sides (define it on the left, x>0)."""
    built = []
    for name, kind, pts, amount, width, sym in ops:
        if kind == "carve":
            built.append((kind, np.asarray(pts[0], float), amount, width, sym))
            _lint(name, "length", 2 * max(amount))
            _lint(name, "width", 2 * min(amount))
            continue
        C = spline(on_surface(f, pts), width / 2) if kind in ("crease", "ridge", "groove") else on_surface(f, pts)
        _lint(name, "width", width)
        _lint(name, "length", float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()))
        reach = width + (amount if kind == "groove" else 0)
        lo, hi = C.min(0) - reach, C.max(0) + reach
        built.append((kind, C, amount, width, sym, lo, hi))

    def g(p):
        d = f(p)
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
    _lint("eye_fissure", "length", float(np.hypot(ex[0] - en[0], ex[1] - en[1])))
    cu, ru = _circle3(en[:2], up[:2], ex[:2])
    cl, rl = _circle3(en[:2], lo[:2], ex[:2])

    def g(p):
        q = mirror_x(p)
        ap = np.maximum(length(q[:, :2] - cu) - ru, length(q[:, :2] - cl) - rl)  # almond in x,y
        ap = smax(ap, np.abs(q[:, 2] - (c[2] + r)) - 0.9 * r, 0.001)            # extrude in front of the eye
        d = smin(f(p), sphere(q, c, r + lid), 0.004)                              # lids wrap the ball
        d = cut(d, ap, 0.0008)                                                    # open the almond
        ball = np.minimum(sphere(q, c, r), sphere(q, c + (0, 0, 0.42 * r), 0.68 * r))  # eyeball + cornea
        return np.minimum(d, ball)
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

    ATTACHED.append(("ear", part, f))
    return lambda p: smin(f(p), part(p), 0.005)


# ---------- senses ----------

def seating(part, host, n=64):
    """Fraction of the part's surface that touches the host (within 1.5 mm). Low = stuck on."""
    s = np.linspace(-0.2, 0.2, n * 2)
    g = np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3)
    g = g[g[:, 0] > 0]  # left side is enough (mirrored)
    surf = g[np.abs(part(g)) < 0.002]
    return float((host(surf) < 0.0015).mean()) if len(surf) else 0.0


def report(mod):
    lines = []
    for name, (part, host) in {n: (pa, h) for n, pa, h in ATTACHED}.items():  # scene may load twice
        frac = seating(part, host)
        lines.append(f"seating {name}: {frac:.0%} of surface in contact" + ("  << stuck-on / floating (attached ears measure ~55-60%)" if frac < 0.40 else ""))
    if LINT:
        lines.append("scale lint: " + "; ".join(dict.fromkeys(LINT)))
    corr = getattr(mod, "FIT_CORRECTIONS", {})
    if corr:
        top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:5]
        lines.append("fit: base was off by " + ", ".join(f"{k} {v * 1000:+.1f}mm" for k, v in top))
    L = getattr(mod, "LANDMARKS", {})
    if L:
        names = list(L)
        d = mod.sdf(np.array([L[k] for k in names]))
        off = [(k, v) for k, v in zip(names, d) if abs(v) > 0.0005 and not k.endswith("_R")]
        lines.append("landmark drift (>0.5mm, +out/-in): " + (", ".join(f"{k} {v * 1000:+.1f}" for k, v in off) or "none"))
    prof = getattr(mod, "PROFILE", [])
    if prof:
        P = np.array([L[k] for k in prof])
        worst = []
        for (a, pa), (b, pb) in zip(zip(prof, P), zip(prof[1:], P[1:])):
            seg = spline(np.array([pa, pb]), 0.001)
            dev = mod.sdf(seg)
            i = int(np.argmax(np.abs(dev)))
            worst.append((abs(dev[i]), f"{a}->{b} {dev[i] * 1000:+.1f}mm"))
        lines.append("profile max dev: " + ", ".join(s for v, s in sorted(worst, reverse=True)[:4]))
    return "\n".join(lines)


SHORT = {"glabella": "g", "nasion": "n", "pronasale": "prn", "subnasale": "sn", "labrale_sup": "ls",
         "stomion": "sto", "labrale_inf": "li", "sublabiale": "sl", "pogonion": "pg", "menton": "me"}


def render_labeled(path, out, center, half, res=320, labels=True):
    mod = load(path)
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
            rel = v - np.asarray(center)
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
    A, B = Image.open(a), Image.open(b)
    m = Image.new("RGB", (A.width + B.width, max(A.height, B.height)), (255, 255, 255))
    m.paste(A, (0, 0))
    m.paste(B, (A.width, 0))
    d = ImageDraw.Draw(m)
    d.text((6, A.height - 14), labels[0], fill=(0, 0, 255))
    d.text((A.width + 6, B.height - 14), labels[1], fill=(0, 0, 255))
    m.save(out)


if __name__ == "__main__":
    scene, out = sys.argv[1], sys.argv[2]
    *c, h = map(float, (sys.argv[3] if len(sys.argv) > 3 else "0,-0.01,0,0.14").split(","))
    t0 = time.time()
    render_labeled(scene, out, c, h)
    import facelab  # the module instance the scene imported (not __main__)
    print(facelab.report(load(scene)))
    print(f"render+report {time.time() - t0:.1f}s, scene {len(open(scene, encoding='utf-8').read())} chars (~{len(open(scene, encoding='utf-8').read()) // 4} tokens)")
