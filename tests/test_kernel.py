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
    assert rep.startswith("WARN loose piece (moon)") and "components=2" in rep


def test_load_resets_registries(tmp_path):
    scene = tmp_path / "s.py"
    scene.write_text("from formcode import *\nTYPICAL.clear()\nsdf = lambda p: sphere(p, (0, 0, 0), 0.3)\n", encoding="utf-8")
    load(scene)
    assert not K.TYPICAL
    K.LINT.append("stale")
    load(tmp_path / "s.py")
    K.reset()
    assert "ear" in K.TYPICAL and not K.LINT


def scene(tmp_path, body):
    path = tmp_path / "s.py"
    path.write_text("from formcode import *\n" + body + "\n", encoding="utf-8")
    return report(load(path))


def test_thin_neck_is_not_loose(tmp_path):
    # a 4 mm wire joins two balls: thinner than the 21 mm report grid, but attached
    rep = scene(tmp_path, "sdf = lambda p: np.minimum(np.minimum(sphere(p, (-0.4, 0, 0), 0.2), sphere(p, (0.4, 0, 0), 0.2)),"
                          " capsule(p, (-0.3, 0, 0), (0.3, 0, 0), 0.004))")
    assert "loose" not in rep and "attached by necks" in rep


def test_kinks_found_on_hard_edges_not_on_smooth_shapes(tmp_path):
    assert scene(tmp_path, "sdf = lambda p: ellipsoid(p, (0, 0, 0), (0.5, 0.3, 0.4))").startswith("warnings: none")
    rep = scene(tmp_path, "sdf = lambda p: box(p, (0, 0, 0), (0.4, 0.3, 0.3))")
    assert "WARN kinks" in rep


def test_kinks_at_intended_edges_are_not_warned(tmp_path):
    rep = scene(tmp_path, "sdf = sculpt(lambda p: sphere(p, (0, 0, 0), 0.5), "
                          "[('slit', 'groove', [(-0.2, 0, 0.46), (0, 0, 0.5), (0.2, 0, 0.46)], 0.01, 0.002, False)])")
    assert "WARN kinks" not in rep and "at intended edges" in rep


def test_front_outline_kinks_name_the_trim(tmp_path):
    rep = scene(tmp_path, "sdf = front_outline(lambda p: sphere(p, (0, 0, 0), 0.6), [(0.5, 0.3), (0.0, 0.45), (-0.5, 0.3)], k=0.001)")
    assert "front_outline trim" in rep


def test_clipping_is_warned(tmp_path):
    assert "WARN clipped at FRAME +x" in scene(tmp_path, "sdf = lambda p: sphere(p, (0.5, 0, 0), 0.6)")
