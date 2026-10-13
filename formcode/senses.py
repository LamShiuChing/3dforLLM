"""Text senses: one report, every check on every run (a sense that is skipped hides defects)."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import kernel
from .kernel import peval, polygon2d, spline
from .render import frame, normals


def fmt(v):
    return ",".join(f"{x:+.2f}" for x in v)


def seating(part, host, n=64, box=None, left=True, surf=None):
    """Fraction of the part's surface that touches the host (within 1.5 mm). Low = stuck on. Works for any part vs
    any host (the engine passes a node and the model without it). The grid is fixed: 2n points across +-0.2 m,
    continued past it. box=(lo, hi): only the grid points inside it (the same result if it holds the part's surface,
    much faster). left: only x>0 (enough for a mirrored part). surf: part_surface(part, ...), if already known."""
    surf = part_surface(part, n, box, left) if surf is None else surf
    return float((peval(host, surf) < 0.0015).mean()) if len(surf) else 0.0


def part_surface(part, n=64, box=None, left=True):
    """The fixed-grid points within 2 mm of the part's surface (seating's sample)."""
    h = 0.4 / (2 * n - 1)
    lo, hi = (np.full(3, -0.2), np.full(3, 0.2)) if box is None else box
    s = [-0.2 + h * np.arange(np.ceil((a + 0.2) / h - 1e-9), np.floor((b + 0.2) / h + 1e-9) + 1) for a, b in zip(lo, hi)]
    g = np.stack(np.meshgrid(*s, indexing="ij"), -1).reshape(-1, 3)
    if left:
        g = g[g[:, 0] > 0]
    return g[np.abs(peval(part, g)) < 0.002]


def mm(v):
    return ",".join(f"{x * 1000:+.0f}" for x in v)


def _grid(c, half, n):
    s = np.linspace(-half, half, n)
    return c + np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3)


def _connected(f, piece, body, tree, h):
    """Recheck on a grid 8x finer around the gap between the piece and the body: are they joined through the inside?"""
    # ponytail: only the box around the facing region (+-2 voxels) is searched; a neck that loops further out is missed
    dist, j = tree.query(piece)
    close = dist < dist.min() + 2 * h
    A, B = piece[close], body[j[close]]
    lo, hi = np.minimum(A.min(0), B.min(0)) - 2 * h, np.maximum(A.max(0), B.max(0)) + 2 * h
    n = np.minimum(np.ceil((hi - lo) / (h / 8)).astype(int) + 1, 96)
    g = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], n[i]) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    lab, _ = ndimage.label((f(g) < 0).reshape(n))

    def near(Q):  # labels in the 3x3x3 nodes around each point (a single node can miss a thin tip)
        out = set()
        for i in np.round((Q - lo) / (hi - lo) * (n - 1)).astype(int):
            out |= set(lab[tuple(slice(max(k - 1, 0), k + 2) for k in i)].ravel())
        return out - {0}
    return bool(near(A) & near(B))


def _fn(f, p, e):
    """f and its unit normal at p, from one call of f (f may run it in threads; same values as render.normals)."""
    E = np.eye(3) * e
    v = f(np.concatenate([p] + [p + x for x in E] + [p - x for x in E])).reshape(7, len(p))
    n = (v[1:4] - v[4:]).T
    return v[0], n / np.maximum(kernel.length(n)[:, None], 1e-9)


def _tangent_turns(f, P, t, span, e, k=7):
    """Normal turn (deg) between neighbours along a short line through each surface point, one line per tangent field
    in t (all in two calls of f): (largest step, total), one row per tangent field."""
    u = np.linspace(-span / 2, span / 2, k)
    Q = (P[None, None] + u[None, :, None, None] * t[:, None]).reshape(-1, 3)
    v, n = _fn(f, Q, e)
    Q = Q - v[:, None] * n  # back onto the surface
    Nq = normals(f, Q, e).reshape(len(t), k, -1, 3)
    steps = np.degrees(np.arccos(np.clip((Nq[:, 1:] * Nq[:, :-1]).sum(-1), -1, 1)))
    return steps.max(1), steps.sum(1)


def kinks(f, g, d, h, n_samples=3000):
    """Surface points where the surface creases sharply (C1 break): one tiny step along the surface carries most of
    the normal's turn. Smooth curvature spreads the turn evenly. Returns kinked points (and the sample count)."""
    shell = g[np.abs(d) < h / 2]
    rng = np.random.default_rng(0)
    if len(shell) > n_samples:
        shell = shell[rng.choice(len(shell), n_samples, replace=False)]
    e = h * 0.02
    P = shell
    for _ in range(3):
        v, n = _fn(f, P, e)
        P = P - v[:, None] * n
    N = normals(f, P, e)
    t1 = np.cross(N, rng.normal(size=N.shape))
    t1 /= np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
    (m1, m2), (s1, s2) = _tangent_turns(f, P, np.stack([t1, np.cross(N, t1)]), h, e)
    mx, tot = np.maximum(m1, m2), np.where(m1 > m2, s1, s2)
    return P[(mx > 4) & (mx / np.maximum(tot, 1e-9) > 0.5)], len(P)


def _kink_warnings(K, h, L, parts):
    """Group kinks (folded to x>=0, so mirrored pairs merge) into clusters and describe where each one is:
    (key, text, [samples, *lo, *hi]) with the unfolded box (for the regression rule, versions.diff)."""
    out = []
    if not len(K):
        return out
    F = K * (-1, 1, 1) * (K[:, :1] < 0) + K * (K[:, :1] >= 0)
    idx = np.floor((F - F.min(0)) / h).astype(int)
    vol = np.zeros(idx.max(0) + 1, bool)
    vol[tuple(idx.T)] = True
    lab, k = ndimage.label(ndimage.binary_dilation(vol, iterations=2))
    which = lab[tuple(idx.T)]
    names = list(L)
    for comp in sorted(range(1, k + 1), key=lambda i: -(which == i).sum()):
        sel = which == comp
        if sel.sum() < 3:
            continue
        pts, raw = F[sel], K[sel]
        lo, hi = pts.min(0), pts.max(0)
        side = "midline" if hi[0] < 1.5 * h else ("both sides" if (raw[:, 0] < 0).any() and (raw[:, 0] > 0).any() else ("left" if raw[0, 0] > 0 else "right"))
        near = ""
        if names:
            V = np.array([L[nm] for nm in names])
            dist = np.linalg.norm(V - pts.mean(0), axis=-1)
            near = f" near {names[int(np.argmin(dist))]}" if dist.min() < 10 * h else ""
        if not near and parts:
            near = " on " + min(parts, key=lambda nm: np.abs(parts[nm](pts)).mean())
        hint = "  (mirror seam: the shape is not smooth across x=0)" if side == "midline" else ""
        if any((np.abs(polygon2d(pts[:, :2], V)) < 2 * h).mean() > 0.5 for V in kernel.TRIMS):
            hint = "  (on the front_outline trim: its polygon corners crease the surface; smooth the outline or raise k)"
        out.append((f"kinks {side}{near}", f"WARN kinks: {sel.sum()} creased samples, {side}{near}, box {mm(lo)}..{mm(hi)}mm{hint}",
                    [int(sel.sum()), *map(float, raw.min(0)), *map(float, raw.max(0))]))
        if len(out) == 3:
            break
    return out


def measure(mod, n=96, cache=None):
    """Every sense as data. warnings: (key, text) pairs, key = kind + subject (for diffs); lines: the measurement text.
    cache: a gridcache.GridCache (the engine's): grid values and part boxes are reused where the build didn't change."""
    c, half = frame(mod)
    g = _grid(c, half, n)
    h = 2 * half / (n - 1)
    vox = h ** 3
    F = lambda q: peval(mod.sdf, q)  # threaded, same values
    key = (*c, half, n)
    d = cache.field(mod, g, key) if cache is not None and hasattr(mod, "STAGES") else F(g)
    warns, lines = [], []
    M = {"bbox": {}, "seating": {}, "kinks": None, "fit": {}, "drift": {}, "profile": {}}
    parts = getattr(mod, "PARTS", {})
    boxes = {}  # (grid key, shape sig) -> (lo, hi, voxels inside): a shape's box only changes when the shape does
    for name, f in [("ALL", None)] + list(parts.items()):
        sig = getattr(f, "sig", None) and (key, f.sig)
        box = cache.parts.get(sig) if sig and cache is not None else None
        if box is None:
            inside = g[(d if f is None else peval(f, g)) < 0]
            box = (inside.min(0), inside.max(0), len(inside)) if len(inside) else (None, None, 0)
        if sig:
            boxes[sig] = box
        lo, hi, vol = box
        if not vol:
            lines.append(f"{name}: empty")
            M["bbox"][name] = None
            continue
        M["bbox"][name] = [round(float(x), 4) for x in (*lo, *hi)]
        lines.append(f"{name}: min={fmt(lo)} max={fmt(hi)} size={fmt(hi - lo)} vol={vol * vox:.3f}")
    if cache is not None:
        cache.parts = boxes
    lab_, k = ndimage.label((d < 0).reshape(n, n, n))
    thin = 0
    if k > 1:  # every piece not connected to the largest one: recheck finer, then warn with the part it belongs to
        sizes = ndimage.sum(np.ones_like(lab_), lab_, range(1, k + 1))
        main = 1 + int(np.argmax(sizes))
        body = g[(lab_ == main).ravel()]
        tree = cKDTree(body)
        for comp in range(1, k + 1):
            if comp == main:
                continue
            pts = g[(lab_ == comp).ravel()]
            if _connected(F, pts, body, tree, h):
                thin += 1
                continue
            owner = min(parts.items(), key=lambda kv: kv[1](pts).mean(), default=("?", None))[0]
            warns.append((f"loose piece ({owner})", f"WARN loose piece ({owner}) at {fmt(pts.mean(0))} vox={len(pts)}: not connected to the body"))
    lines.append(f"components={k}" + (f" ({thin} attached by necks thinner than {h * 1000:.0f}mm)" if thin else ""))
    band = 0.03 * half
    shell = np.abs(d) < band
    m = g[shell].copy()
    m[:, 0] *= -1
    over = np.abs(g[d < 0] - c) > half - band
    clipped = bool(over.any())
    sym = np.abs(F(m) - d[shell]).mean()
    M.update(components=int(k), thin_necks=thin, symmetry_err=float(sym), clipped=clipped)
    lines.append(f"symmetry_err(x)={sym:.4f}  clipped_at_bound={clipped}")
    if clipped:
        P = g[d < 0][over.any(1)] - c
        sides = " ".join(sorted({("-+"[int(v > 0)] + "xyz"[i]) for i in range(3) for v in P[np.abs(P[:, i]) > half - band, i]}))
        warns.append((f"clipped at FRAME {sides}", f"WARN clipped at FRAME {sides}: the model extends past it (open mesh); grow or move FRAME"))

    surfs = {}  # part sig -> its surface sample: it only changes when the part does
    for name, (part, host) in {nm: (pa, hh) for nm, pa, hh in kernel.ATTACHED}.items():
        sig = getattr(part, "sig", None)
        surf = cache.surfs.get(sig) if sig and cache is not None else None
        surfs[sig] = surf = part_surface(part) if surf is None else surf
        frac = seating(part, host, surf=surf)
        M["seating"][name] = frac
        lines.append(f"seating {name}: {frac:.0%} of surface in contact")
        if frac < 0.40:
            warns.append((f"seating {name}", f"WARN seating {name}: only {frac:.0%} in contact: stuck on / floating (attached ears measure ~55-60%)"))
    if cache is not None:
        cache.surfs = surfs
    warns += [("scale: " + " ".join(x.split()[:2]), f"WARN scale: {x}") for x in dict.fromkeys(kernel.LINT)]

    K, total = kinks(F, g, d, h)
    if len(K):
        R = np.zeros(len(K), bool)
        for P_, r in kernel.EDGES:
            R |= np.linalg.norm(K[:, None] - P_[None], axis=-1).min(1) < r
        M["kinks"] = {"creased": len(K), "samples": total, "intended": int(R.sum())}
        lines.append(f"kinks: {len(K)} of {total} surface samples creased ({R.sum()} at intended edges)")
        if kernel.DETAIL:
            lines[-1] += f"; {', '.join(sorted(kernel.DETAIL))} detail: kinks expected, not warned"
        else:
            kw = _kink_warnings(K[~R], h, getattr(mod, "LANDMARKS", {}), getattr(mod, "PARTS", {}))
            warns += [(k, t) for k, t, _ in kw]
            M["kink_clusters"] = {k: c for k, _, c in reversed(kw)}  # the largest cluster per key
    corr = getattr(mod, "FIT_CORRECTIONS", {})
    if corr:
        M["fit"] = {k: float(v) * 1000 for k, v in corr.items()}
        top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:5]
        lines.append("fit: base was off by " + ", ".join(f"{k} {v * 1000:+.1f}mm" for k, v in top))
    L = getattr(mod, "LANDMARKS", {})
    if L:
        names = list(L)
        dl = mod.sdf(np.array([L[k] for k in names]))
        M["drift"] = {k: float(v) * 1000 for k, v in zip(names, dl) if not k.endswith("_R")}
        off = [(k, v) for k, v in zip(names, dl) if abs(v) > 0.0005 and not k.endswith("_R")]
        lines.append("landmark drift (>0.5mm, +out/-in): " + (", ".join(f"{k} {v * 1000:+.1f}" for k, v in off) or "none"))
    prof = getattr(mod, "PROFILE", [])
    if prof:
        P = np.array([L[k] for k in prof])
        worst = []
        for (a, pa), (b, pb) in zip(zip(prof, P), zip(prof[1:], P[1:])):
            dev = mod.sdf(spline(np.array([pa, pb]), 0.001))
            i = int(np.argmax(np.abs(dev)))
            worst.append((abs(dev[i]), f"{a}->{b} {dev[i] * 1000:+.1f}mm"))
            M["profile"][f"{a}->{b}"] = float(dev[i]) * 1000
        lines.append("profile max dev: " + ", ".join(s for v, s in sorted(worst, reverse=True)[:4]))
    return dict(M, warnings=warns, lines=lines)


def text(m):
    """The report: warnings first, then measurements."""
    return "\n".join(([t for _, t in m["warnings"]] or ["warnings: none"]) + ["--"] + m["lines"])


def report(mod, n=96):
    return text(measure(mod, n))
