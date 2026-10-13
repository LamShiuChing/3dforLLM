"""The measure grid's values, cached per build stage (PROTOCOL §9): after an edit only the grid points a stage has to
redo are evaluated, every other value is the cached one, bit for bit (so a cached measure equals a fresh one).
A stage (engine.build -> mod.STAGES) is (key, sig, apply(v, p), elems): apply maps the previous stage's values v at
points p to its own, sig says what the stage is. A stage is redone where its input changed (compared bit for bit)
and, if its sig changed, where it can act: the old and new boxes of its changed elements (strokes), else everywhere."""
import numpy as np

from .kernel import peval


def _run(apply, v, p):
    """apply(v, p) in threads (peval): (p, v) travel together as one (N, 4) array."""
    if v is None:
        return peval(lambda q: apply(None, q), p)
    return peval(lambda q: apply(np.ascontiguousarray(q[:, 3]), np.ascontiguousarray(q[:, :3])), np.column_stack([p, v]))


def _boxes(old, new, g):
    """Grid points a changed stage can act on: the boxes of its changed elements, old and new. None = anywhere.
    g is a lattice (senses._grid), so a box is an index range per axis."""
    if old is None or new is None or [e[0] for e in old] != [e[0] for e in new]:
        return None
    n = round(len(g) ** (1 / 3))
    G = g.reshape(n, n, n, 3)
    axes = (G[:, 0, 0, 0], G[0, :, 0, 1], G[0, 0, :, 2])
    m = np.zeros((n, n, n), bool)
    for (_, s0, b0), (_, s1, b1) in zip(old, new):
        if s0 != s1:
            for b in (b0, b1):
                if b is None:
                    return None
                lo, hi, sym = b  # sym: the op acts at |x| (kernel.mirror_x)
                m[np.ix_(*[np.flatnonzero((v > lo[i]) & (v < hi[i])) for i, v in
                           enumerate((np.abs(axes[0]) if sym else axes[0], *axes[1:]))])] = True
    return m.ravel()


class GridCache:
    """One per engine: the last grid it measured, per stage, the parts' boxes on it and the seating samples."""

    def __init__(self):
        self.key, self.stages, self.parts, self.surfs = None, [], {}, {}

    def field(self, mod, g, key):
        """mod.sdf on grid g (key: its center, half and n), redoing only what changed since the last call."""
        same = key == self.key and [s[0] for s in self.stages] == [s[0] for s in mod.STAGES]
        out, v, changed = [], None, np.zeros(len(g), bool)
        for i, (k, sig, apply, elems) in enumerate(mod.STAGES):
            if not same:
                w = _run(apply, v, g)
            else:
                w0 = self.stages[i][3]
                m = changed
                if sig != self.stages[i][1]:
                    r = _boxes(self.stages[i][2], elems, g)
                    m = np.ones(len(g), bool) if r is None else m | r
                if m.all():
                    w = _run(apply, v, g)
                elif m.any():
                    w = w0.copy()
                    w[m] = _run(apply, None if v is None else v[m], g[m])
                else:
                    w = w0
                changed = w != w0
            out.append((k, sig, elems, w))
            v = w
        self.key, self.stages = key, out
        return v
