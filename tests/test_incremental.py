"""Step D: the engine's cached measure (gridcache) must equal a from-scratch one after every edit, bit for bit."""
import numpy as np

from formcode.engine import Scene, parse
from formcode.gridcache import GridCache
from formcode.kernel import peval
from formcode.render import frame
from formcode.senses import _grid, measure

SCENE = """new('s')
add('head', 'ellipsoid', at=(0, 0, 0), size=(0.075, 0.1, 0.09))
add('cheek', 'ellipsoid', at=(0.04, -0.03, 0.05), size=(0.018, 0.02, 0.016), blend=0.02, mirror=True)
add('nose', 'sphere', at=on('body', x=0, y=-0.01), size=0.012)
mark('nasion', on('head', x=0, y=0.02))
mark('chin', on('head', x=0, y=-0.08))
curve('profile', through=['nasion', 'chin'])
stroke('fold', 'crease', along=[on('body', x=0.02, y=-0.03), on('body', x=0.03, y=-0.06)], amount=0.0012, width=0.009, mirror=True)
stroke('pit', 'carve', along=[(0.0, -0.06, 0.09)], amount=(0.004, 0.002, 0.004), width=0.002)
add('eye', 'eye', center=(0.03, 0.01), r=0.011, en=(0.017, 0.01), ex=(0.044, 0.012), up=(0.03, 0.016), lo=(0.031, 0.005))
add('ear', 'ear', tragion=(-0.02, -0.005))
"""


def edits(rng):
    """One random edit of each kind the engine sees: strokes, shapes (frame kept or moved), marks, eye, ear, add/delete."""
    u = lambda a, b: round(float(rng.uniform(a, b)), 4)
    return [f'set("fold", amount={u(0.0005, 0.002)})',
            f'set("fold", width={u(0.005, 0.012)})',
            f'move("cheek", at=({u(0.035, 0.045)}, {u(-0.035, -0.025)}, {u(0.045, 0.055)}))',
            f'set("nose", size={u(0.008, 0.016)})',
            f'set("pit", amount=({u(0.003, 0.005)}, 0.002, 0.004))',
            f'move("chin", at=on("head", x=0, y={u(-0.085, -0.07)}))',
            f'set("ear", splay={u(-30, 20)}, flare={u(0.002, 0.008)})',
            f'set("eye", recess={u(0.0003, 0.001)})',
            f'set("head", size=({u(0.07, 0.08)}, 0.1, 0.09))',
            'add("bump", "sphere", at=on("body", x=-0.045, y=0.02), size=0.005, blend=0.004)',
            'delete("bump")']


def test_cached_measure_equals_fresh_after_every_edit():
    s, cache, rng = Scene(), GridCache(), np.random.default_rng(7)
    s.replay(SCENE, "scene")
    seen = set()
    for step in range(12):
        if step:
            pool = [c for c in edits(rng) if not (c.startswith("delete") and "bump" not in s.nodes)
                    and not (c.startswith("add") and "bump" in s.nodes)]
            cmd = pool[rng.integers(len(pool))]
            seen.add(cmd.split(",")[0])  # command + node
            s.call(*parse(cmd))
        mod = s.build()
        m_inc = measure(mod, cache=cache)
        c, half = frame(mod)
        assert np.array_equal(cache.stages[-1][3], peval(mod.sdf, _grid(c, half, 96))), f"step {step}: grid values differ"
        assert m_inc == measure(mod), f"step {step}: cached measure differs from a fresh one"
    assert len(seen) >= 6, seen  # the sequence really mixed edit kinds
