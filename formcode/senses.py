"""Text senses: one report, every check on every run (a sense that is skipped hides defects)."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import kernel
from .kernel import polygon2d, spline
from .render import frame, normals


def fmt(v):
    return ",".join(f"{x:+.2f}" for x in v)


def seating(part, host, n=64):
    """Fraction of the part's surface that touches the host (within 1.5 mm). Low = stuck on."""
    s = np.linspace(-0.2, 0.2, n * 2)
    g = np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3)
    g = g[g[:, 0] > 0]  # left side is enough (mirrored)
    surf = g[np.abs(part(g)) < 0.002]
    return float((host(surf) < 0.0015).mean()) if len(surf) else 0.0


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


def _tangent_turns(f, P, N, t, span, e, k=7):
    """Normal turn (deg) between neighbours along a short line through each surface point: (largest step, total)."""
    u = np.linspace(-span / 2, span / 2, k)
    Q = (P[None] + u[:, None, None] * t[None]).reshape(-1, 3)
    Q = Q - f(Q)[:, None] * normals(f, Q, e)  # back onto the surface
    Nq = normals(f, Q, e).reshape(k, -1, 3)
    steps = np.degrees(np.arccos(np.clip((Nq[1:] * Nq[:-1]).sum(-1), -1, 1)))
    return steps.max(0), steps.sum(0)


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
        P = P - f(P)[:, None] * normals(f, P, e)
    N = normals(f, P, e)
    t1 = np.cross(N, rng.normal(size=N.shape))
    t1 /= np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
    m1, s1 = _tangent_turns(f, P, N, t1, h, e)
    m2, s2 = _tangent_turns(f, P, N, np.cross(N, t1), h, e)
    mx, tot = np.maximum(m1, m2), np.where(m1 > m2, s1, s2)
    return P[(mx > 4) & (mx / np.maximum(tot, 1e-9) > 0.5)], len(P)


def _kink_warnings(K, h, L, parts):
    """Group kinks (folded to x>=0, so mirrored pairs merge) into clusters and describe where each one is."""
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
        out.append(f"WARN kinks: {sel.sum()} creased samples, {side}{near}, box {mm(lo)}..{mm(hi)}mm{hint}")
        if len(out) == 3:
            break
    return out


def report(mod, n=96):
    c, half = frame(mod)
    g = _grid(c, half, n)
    h = 2 * half / (n - 1)
    vox = h ** 3
    d = mod.sdf(g)
    warns, lines = [], []
    parts = getattr(mod, "PARTS", {})
    for name, f in [("ALL", None)] + list(parts.items()):
        inside = g[(d if f is None else f(g)) < 0]
        if not len(inside):
            lines.append(f"{name}: empty")
            continue
        lo, hi = inside.min(0), inside.max(0)
        lines.append(f"{name}: min={fmt(lo)} max={fmt(hi)} size={fmt(hi - lo)} vol={len(inside) * vox:.3f}")
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
            if _connected(mod.sdf, pts, body, tree, h):
                thin += 1
                continue
            owner = min(parts.items(), key=lambda kv: kv[1](pts).mean(), default=("?", None))[0]
            warns.append(f"WARN loose piece ({owner}) at {fmt(pts.mean(0))} vox={len(pts)}: not connected to the body")
    lines.append(f"components={k}" + (f" ({thin} attached by necks thinner than {h * 1000:.0f}mm)" if thin else ""))
    band = 0.03 * half
    shell = np.abs(d) < band
    m = g[shell].copy()
    m[:, 0] *= -1
    over = np.abs(g[d < 0] - c) > half - band
    clipped = bool(over.any())
    lines.append(f"symmetry_err(x)={np.abs(mod.sdf(m) - d[shell]).mean():.4f}  clipped_at_bound={clipped}")
    if clipped:
        P = g[d < 0][over.any(1)] - c
        sides = sorted({("-+"[int(v > 0)] + "xyz"[i]) for i in range(3) for v in P[np.abs(P[:, i]) > half - band, i]})
        warns.append(f"WARN clipped at FRAME {' '.join(sides)}: the model extends past it (open mesh); grow or move FRAME")

    for name, (part, host) in {nm: (pa, hh) for nm, pa, hh in kernel.ATTACHED}.items():
        frac = seating(part, host)
        lines.append(f"seating {name}: {frac:.0%} of surface in contact")
        if frac < 0.40:
            warns.append(f"WARN seating {name}: only {frac:.0%} in contact: stuck on / floating (attached ears measure ~55-60%)")
    warns += [f"WARN scale: {x}" for x in dict.fromkeys(kernel.LINT)]

    K, total = kinks(mod.sdf, g, d, h)
    if len(K):
        R = np.zeros(len(K), bool)
        for P_, r in kernel.EDGES:
            R |= np.linalg.norm(K[:, None] - P_[None], axis=-1).min(1) < r
        lines.append(f"kinks: {len(K)} of {total} surface samples creased ({R.sum()} at intended edges)")
        if kernel.DETAIL:
            lines[-1] += f"; {', '.join(sorted(kernel.DETAIL))} detail: kinks expected, not warned"
        else:
            warns += _kink_warnings(K[~R], h, getattr(mod, "LANDMARKS", {}), getattr(mod, "PARTS", {}))
    corr = getattr(mod, "FIT_CORRECTIONS", {})
    if corr:
        top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:5]
        lines.append("fit: base was off by " + ", ".join(f"{k} {v * 1000:+.1f}mm" for k, v in top))
    L = getattr(mod, "LANDMARKS", {})
    if L:
        names = list(L)
        dl = mod.sdf(np.array([L[k] for k in names]))
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
        lines.append("profile max dev: " + ", ".join(s for v, s in sorted(worst, reverse=True)[:4]))
    return "\n".join((warns or ["warnings: none"]) + ["--"] + lines)
