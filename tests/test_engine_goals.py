import pytest

from formcode.session import Engine
from formcode.versions import diff

BALL = 'add("ball", "sphere", at=(0, 0, 0), size=0.05)'
NOSE = 'add("nose", "sphere", at=on("ball", x=0, y=0), size=0.01)'
HEAD = 'add("head", "ellipsoid", at=(0, 0, 0), size=(0.075, 0.1, 0.09))'


def test_seat_attaches_a_loose_sphere():
    e = Engine()
    e.do(BALL)
    assert e.do('add("bud", "sphere", at=(0.08, 0.01, 0), size=0.012)').startswith("REVERTED")
    e.do("keep()")
    assert "loose piece (bud)" in e.do("report()")
    out = e.do('seat("bud")').splitlines()
    assert out[0].startswith("ok seat bud: moved ") and "-WARN loose piece (bud)" in out[1]
    assert int(out[0].rsplit("contact ", 1)[1].rstrip("%")) >= 10
    assert "loose piece" not in e.do("report()")


def test_seat_ear_from_a_bad_splay():
    e = Engine()
    e.do(HEAD + '\nadd("ear", "ear", tragion=(-0.02, -0.005), splay=-35)')
    assert e.metrics["seating"]["ear"] < 0.4
    out = e.do('seat("ear")').splitlines()  # default: just attached (>= 45%), the angle changed as little as possible
    assert out[0].startswith("ok seat ear: ") and "-WARN seating ear" in out[1]
    assert 0.45 <= e.metrics["seating"]["ear"] < 0.55
    assert e.scene.nodes["ear"][1]["splay"] < 12, "seat swung the ear further than needed"
    e2 = Engine()
    e2.do(HEAD + '\nadd("ear", "ear", tragion=(-0.02, -0.005), splay=-35)')
    e2.do('seat("ear", contact=0.57)')  # explicit target: hit it
    assert 0.55 <= e2.metrics["seating"]["ear"] <= 0.59
    assert "no seating measure" in Engine().do(HEAD + '\nadd("eye", "eye", center=(0.03, 0.0), r=0.012, en=(0.017, 0), '
                                               'ex=(0.045, 0), up=(0.03, 0.005), lo=(0.03, -0.005))\nseat("eye")')


def test_solve_width_and_unreachable_target():
    e = Engine()
    e.do(BALL)
    out = e.do('solve("ball.size", "width ball", 0.104)')
    assert out.startswith("ok solve ball.size 0.05->")
    assert abs(float(e.do('measure("ball")').split()[1].split("x")[0]) - 104) <= 0.5
    out = e.do('solve("ball.size", "width ball", 9.0)')
    assert "not reachable" in out and "closest" in out and "Nothing applied" in out
    assert 'did you mean "size"?' in e.do('solve("ball.sise", "width ball", 0.1)')
    assert "unknown measure" in e.do('solve("ball.size", "girth ball", 0.1)')
    assert "unknown measure" in e.do('solve("ball.size", "girth ball", 0.1)')


def test_fix_search_offers_seat_and_fix_applies_it():
    e = Engine()
    e.do(BALL + "\n" + NOSE)
    out = e.do('move("nose", at=(0, 0, 0.075))').splitlines()
    assert out[0] == "REVERTED v002: +WARN loose piece (nose)"
    assert out[2].startswith('fix 1: seat("nose") -> contact ') and out[2].endswith("warnings: none")
    assert out[-1] == "fix(1) to apply | keep() to accept the change anyway"
    out = e.do("fix(9)")
    assert out.startswith("error line 1: fix(9): no fix 9") and "fix(1)" in out
    out = e.do("fix(1)").splitlines()
    assert out[0].startswith("ok seat nose: moved ") and out[-1].startswith("warnings: none | v003")
    assert e.do("report()").startswith("warnings: none")
    assert "the scene changed" in e.do("fix(1)")


FLICKER = (BALL + '\nadd("bud", "sphere", at=on("ball", x=0.03, y=0.02), size=0.006, op="separate")\n'
           'add("nose", "sphere", at=on("ball", x=-0.02, y=-0.03), size=0.01)')


@pytest.mark.parametrize("size", [0.008, 0.016])
def test_far_kink_flicker_is_not_a_regression(size):
    """Resizing the nose reshuffles the kink samples: a 3-sample cluster at the bud (0.008), or a 6-sample one
    outside the changed region (0.016), shows up. Shown as a warning, but not reverted (step B reverted these)."""
    e = Engine()
    e.do(FLICKER)
    out = e.do(f'set("nose", size={size})').splitlines()
    assert "+WARN kinks left on" in out[1] and out[2].startswith("1 warning | v002")


def test_kink_cluster_regression_rule():
    old = {"warnings": [], "profile": {}, "drift": {}, "seating": {}}
    new = dict(old, warnings=[("kinks left", "WARN kinks: ...")], kink_clusters={"kinks left": [6, 0, 0, 0, 0.01, 0.01, 0.01]})
    assert diff(old, new)[1] == ["+WARN kinks left"]                                   # big enough, no region: counts
    assert diff(old, new, region=[((0.005, 0, 0), (0.1, 0.1, 0.1))])[1] == ["+WARN kinks left"]  # inside the change
    assert diff(old, new, region=[((0.05, 0, 0), (0.1, 0.1, 0.1))])[1] == []           # far from the change
    new["kink_clusters"]["kinks left"][0] = 4
    assert diff(old, new)[1] == [] and diff(old, new)[0] == ["+WARN kinks left"]        # tiny: shown, not a regression


def test_paths_stay_in_the_workspace(tmp_path):
    e = Engine(root=tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    e.do(BALL)
    for bad in ['save("../x.fc")', f'save("{(tmp_path / "x.fc").as_posix()}")', f'open("{(tmp_path / "x.fc").as_posix()}")']:
        out = e.do(bad)
        assert "path outside the workspace" in out and "Nothing applied" in out
    assert "use a .glb path" in e.do('export("x.exe")') and "use a .fc path" in e.do('save("x.py")')
    assert e.do('save("scenes/s.fc")') == "ok saved scenes/s.fc" and (tmp_path / "ws" / "scenes" / "s.fc").is_file()
    assert not (tmp_path / "x.fc").exists()
