"""Text senses: one report, every check on every run (a sense that is skipped hides defects)."""
import numpy as np
from scipy import ndimage

from . import kernel
from .kernel import spline
from .render import frame


def fmt(v):
    return ",".join(f"{x:+.2f}" for x in v)


def seating(part, host, n=64):
    """Fraction of the part's surface that touches the host (within 1.5 mm). Low = stuck on."""
    s = np.linspace(-0.2, 0.2, n * 2)
    g = np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3)
    g = g[g[:, 0] > 0]  # left side is enough (mirrored)
    surf = g[np.abs(part(g)) < 0.002]
    return float((host(surf) < 0.0015).mean()) if len(surf) else 0.0


def report(mod, n=96):
    c, half = frame(mod)
    s = np.linspace(-half, half, n)
    g = c + np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3)
    vox = (2 * half / (n - 1)) ** 3
    d = mod.sdf(g)
    lines = []
    parts = getattr(mod, "PARTS", {})
    for name, f in [("ALL", None)] + list(parts.items()):
        inside = g[(d if f is None else f(g)) < 0]
        if not len(inside):
            lines.append(f"{name}: empty")
            continue
        lo, hi = inside.min(0), inside.max(0)
        lines.append(f"{name}: min={fmt(lo)} max={fmt(hi)} size={fmt(hi - lo)} vol={len(inside) * vox:.3f}")
    lab_, k = ndimage.label((d < 0).reshape(n, n, n))
    if k > 1:  # report every piece not connected to the largest one, with the part it belongs to
        sizes = ndimage.sum(np.ones_like(lab_), lab_, range(1, k + 1))
        main = 1 + int(np.argmax(sizes))
        for comp in range(1, k + 1):
            if comp == main:
                continue
            pts = g[(lab_ == comp).ravel()]
            if len(pts) < 4:  # thinner than the grid (spike tips): sampling noise
                continue
            owner = min(parts.items(), key=lambda kv: kv[1](pts).mean(), default=("?", None))[0]
            lines.append(f"FLOATING piece ({owner}) at {fmt(pts.mean(0))} vox={len(pts)}")
    lines.append(f"components={k}")
    band = 0.03 * half
    shell = np.abs(d) < band
    m = g[shell].copy()
    m[:, 0] *= -1
    clipped = bool((np.abs(g[d < 0] - c) > half - band).any())
    lines.append(f"symmetry_err(x)={np.abs(mod.sdf(m) - d[shell]).mean():.4f}  clipped_at_bound={clipped}")

    for name, (part, host) in {nm: (pa, h) for nm, pa, h in kernel.ATTACHED}.items():
        frac = seating(part, host)
        lines.append(f"seating {name}: {frac:.0%} of surface in contact" + ("  << stuck-on / floating (attached ears measure ~55-60%)" if frac < 0.40 else ""))
    if kernel.LINT:
        lines.append("scale lint: " + "; ".join(dict.fromkeys(kernel.LINT)))
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
    return "\n".join(lines)
