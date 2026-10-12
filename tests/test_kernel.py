import numpy as np

import formcode.kernel as K
from formcode.render import load
from formcode.senses import report


def test_sphere_and_blend():
    p = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    assert np.allclose(K.sphere(p, (0, 0, 0), 1), [-1, 1])
    assert (K.smin(np.array([0.1]), np.array([0.1]), 0.1) < 0.1).all()  # a smooth union bulges


def test_fit_passes_through_targets():
    base = lambda p: K.sphere(p, (0, 0, 0), 0.1)
    targets = [(0.12, 0, 0), (0, 0.09, 0), (0, 0, 0.11)]
    f, d0 = K.fit(base, targets, radius=0.05)
    assert np.allclose(f(np.array(targets)), 0, atol=1e-5)
    assert np.allclose(d0, [0.02, -0.01, 0.01])


def test_cast_and_snap_land_on_surface():
    f = lambda p: K.sphere(p, (0, 0, 0), 0.1)
    hit = K.cast(f, (0, 0, 0.5), (0, 0, -1))[0]
    assert abs(hit[2] - 0.1) < 1e-4
    assert np.allclose(f(K.on_surface(f, [(0.2, 0.1, 0)])), 0, atol=1e-6)


def test_floating_part_is_reported(tmp_path):
    scene = tmp_path / "s.py"
    scene.write_text("from formcode import *\n"
                     "body = lambda p: sphere(p, (0, 0, 0), 0.3)\n"
                     "moon = lambda p: sphere(p, (0.7, 0, 0), 0.1)\n"
                     "sdf = lambda p: np.minimum(body(p), moon(p))\n"
                     "PARTS = {'body': body, 'moon': moon}\n", encoding="utf-8")
    rep = report(load(scene))
    assert "FLOATING piece (moon)" in rep and "components=2" in rep


def test_load_resets_registries(tmp_path):
    scene = tmp_path / "s.py"
    scene.write_text("from formcode import *\nTYPICAL.clear()\nsdf = lambda p: sphere(p, (0, 0, 0), 0.3)\n", encoding="utf-8")
    load(scene)
    assert not K.TYPICAL
    K.LINT.append("stale")
    load(tmp_path / "s.py")
    K.reset()
    assert "ear" in K.TYPICAL and not K.LINT
