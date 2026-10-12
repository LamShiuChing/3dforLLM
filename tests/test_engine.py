import io
from pathlib import Path

import numpy as np
import pytest

import formcode.kernel as K
from formcode.engine import Engine, parse, FcError, repl
from formcode.render import frame, render_view

ROOT = Path(__file__).resolve().parents[1]
P = np.random.default_rng(0).uniform(-0.15, 0.15, (500, 3))
HEAD = 'add("head", "ellipsoid", at=(0, 0, 0), size=(0.08, 0.1, 0.09))'


@pytest.mark.parametrize("bad", ['__import__("os")', 'os.system("x")', 'add("a", "sphere", at=(0, 0, x + 1), size=0.1)',
                                 'add("a", "sphere", at=(0, 0, 0), size=lambda: 1)', 'add("a", "sphere", size=2 * 0.1)',
                                 'add("a", "sphere", at=__import__("os"))', 'add(*x)'])
def test_parse_rejects_anything_but_literals(bad):
    with pytest.raises(FcError):
        parse(bad)


def test_parse_literals_and_helpers():
    cmd, args, kw = parse('stroke("s", "crease", along=[on("body", x=-0.01, y=0.02, depth=-0.002), rel("n", 0, -1e-3, 0)], '
                          'amount=0.001, width=4e-3, mirror=True)')
    assert cmd == "stroke" and args == ["s", "crease"] and kw["mirror"] is True and kw["width"] == 0.004
    assert repr(kw["along"][0]) == "on('body', x=-0.01, y=0.02, depth=-0.002)" and repr(kw["along"][1]) == "rel('n', 0, -0.001, 0)"


def test_batch_is_atomic():
    e = Engine()
    out = e.do(HEAD + '\nadd("b", "sphere", at=__import__("os"), size=0.1)')
    assert out.startswith("error line 2:") and out.endswith("Nothing applied.")
    assert not e.scene.nodes and e.model is None
    out = e.do(HEAD + '\nmark("m", on("head", x=0.5, y=0))')  # a build error (the ray misses) is atomic too
    assert out.startswith("error line 2:") and "missed" in out and not e.scene.nodes


def test_errors_name_the_fix():
    e = Engine()
    e.do(HEAD)
    assert 'did you mean "head"?' in e.do('set("haed", size=0.1)')
    assert 'did you mean "add"?' in e.do('ad("x", "sphere", at=(0, 0, 0), size=0.1)')
    out = e.do('set("head", sise=0.1)')
    assert 'did you mean "size"?' in out and "valid: at, size" in out
    assert "usage: mark(name, at, pin=True)" in e.do('mark("m")')


def test_commands_build_the_same_sdf_as_kernel_code():
    e = Engine()
    out = e.do(HEAD + '\nadd("arm", "capsule", a=(0.05, 0, 0), b=(0.12, -0.05, 0.01), r=0.02, r2=0.015, blend=0.02, mirror=True)')
    assert out.startswith("ok add head (ellipsoid) | ok add arm (capsule, mirror)") and "warnings:" in out
    want = K.smin(K.ellipsoid(P, (0, 0, 0), (0.08, 0.1, 0.09)), K.capsule(K.mirror_x(P), (0.05, 0, 0), (0.12, -0.05, 0.01), 0.02, 0.015), 0.02)
    assert np.allclose(e.model.sdf(P), want)
    c, half = frame(e.model)  # auto-frame holds the whole model
    assert np.all(np.abs(np.array([[0.135, -0.05, 0.01], [-0.135, -0.05, 0.01], [0, 0.1, 0]]) - c) < half)


def test_on_lands_on_the_skin_and_depth_goes_inside():
    e = Engine()
    e.do('add("ball", "sphere", at=(0, 0, 0), size=0.1)\n'
         'mark("a", on("body", x=0.03, y=0.02), pin=False)\n'
         'mark("b", on("ball", x=0.03, y=0.02, depth=-0.002), pin=False)\n'
         'mark("c", on("ball", y=0.01, z=-0.02, side="L"), pin=False)')
    L = e.model.LANDMARKS
    assert abs(e.model.sdf(L["a"][None])[0]) < 1e-4 and L["a"][2] > 0
    assert abs(e.model.sdf(L["b"][None])[0] + 0.002) < 1e-4
    assert abs(e.model.sdf(L["c"][None])[0]) < 1e-4 and L["c"][0] > 0


def test_absolute_placements_are_counted():
    e = Engine()
    e.do('add("ball", "sphere", at=(0, 0, 0), size=0.1)')
    assert e.scene.absolute_placements() == 1
    e.do('mark("a", on("body", x=0.03, y=0.02))')
    assert e.scene.absolute_placements() == 1


def test_save_open_round_trip(tmp_path):
    e = Engine()
    path = (tmp_path / "s.fc").as_posix()
    out = e.do(HEAD + '\nadd("nose", "sphere", at=on("body", x=0, y=-0.01), size=0.015, blend=0.01)\n'
               'mark("ntip", on("nose", x=0, y=-0.01))\n'
               'mark("tip_L", rel("ntip", 0.01, 0, -0.004))\n'
               'stroke("fold", "crease", along=[on("body", x=0.03, y=-0.03), on("body", x=0.04, y=-0.06)], amount=0.001, width=0.006, mirror=True)\n'
               f'rename("nose", "snout")\nsave("{path}")')
    assert "ok rename nose -> snout" in out and "warnings" in out
    assert e.scene.nodes["ntip"][1]["at"].p["host"] == "snout"  # rename follows references
    e2 = Engine()
    assert e2.do(f'open("{path}")').startswith("ok opened untitled (5 nodes)")
    assert np.allclose(e.model.sdf(P), e2.model.sdf(P)) and e.do("tree()") == e2.do("tree()")
    assert "tip_R" in e2.model.LANDMARKS
    assert abs(e2.model.sdf(e2.model.LANDMARKS["tip_R"][None])[0]) < 1e-5  # pinned marks are fitted (and mirrored)
    c, half = frame(e2.model)
    name, img = render_view((path, "front", 16, c, half))  # render.load builds .fc scripts
    assert img.shape == (16, 16) and img.min() < 200


def test_repl(capsys):
    buf = io.StringIO()
    repl([HEAD + "\\\n", 'mark("m", on("body", x=0, y=0))\n', "tree()\n"], buf)
    out = buf.getvalue().splitlines()
    assert out[0] == "ok add head (ellipsoid) | ok mark m" and out[2] == "untitled: 2 nodes"


def test_open_legacy_face():
    e = Engine()
    out = e.do(f'open("{(ROOT / "examples" / "face.py").as_posix()}")')
    assert out == "ok opened face (1 node: face, code)\nwarnings: none"
    assert e.do("tree()") == "face: 1 node\n  face code"
    assert "pronasale" in e.model.LANDMARKS and e.model.PROFILE[0] == "trichion"


def test_non_finite_numbers_rejected():
    from formcode.engine import Engine
    r = Engine().do('add("x", "sphere", at=(0, 0, 0), size=1e999)')
    assert "not a finite number" in r and "Nothing applied" in r
