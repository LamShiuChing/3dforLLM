import json

import pytest

from formcode import versions

BALL = "from formcode import *\nbody = lambda p: sphere(p, (0, 0, 0), 0.3)\n"
MOON = BALL + "moon = lambda p: sphere(p, (0.7, 0, 0), 0.1)\nPARTS = {'body': body, 'moon': moon}\n"


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setattr(versions, "render", lambda *a, **k: None)  # the report is what's versioned; skip the slow render
    path = tmp_path / "s.py"

    def run(src=None, keep=False):
        if src is not None:
            path.write_text(src + "sdf = " + ("lambda p: np.minimum(body(p), moon(p))" if "moon" in src else "body") + "\n",
                            encoding="utf-8")
        return versions.run(path, tmp_path / "out.png", keep=keep)
    run.path, run.runs = path, tmp_path / "runs" / "s"
    return run


def test_first_run_is_baseline_and_unchanged_rerun_adds_nothing(scene):
    out = scene(BALL)
    assert out.splitlines()[-1].startswith("run v001: ~") and (scene.runs / "v001.py").exists()
    assert json.loads((scene.runs / "v001.json").read_text(encoding="utf-8"))["status"] == "baseline"
    assert "unchanged" in scene()
    assert sorted(p.name for p in scene.runs.iterdir()) == ["v001.json", "v001.py"]


def test_regression_is_reverted_and_keep_after_revert_accepts_it(scene):
    scene(BALL)
    v1 = scene.path.read_bytes()
    out = scene(MOON)
    assert out.startswith("REGRESSED vs v001") and "+WARN loose piece (moon)" in out.splitlines()[1]
    assert scene.path.read_bytes() == v1 and (scene.runs / "v002-rejected.py").exists()
    out = scene(keep=True)  # scene untouched since the revert: --keep brings the rejected version back
    assert not out.startswith("REGRESSED") and scene.path.read_bytes() == (scene.runs / "v002-rejected.py").read_bytes()
    assert (scene.runs / "v003.py").exists()


def test_keep_accepts_a_regression(scene):
    scene(BALL)
    out = scene(MOON, keep=True)
    assert not out.startswith("REGRESSED") and "vs v001: +WARN loose piece (moon)" in out
    assert "moon" in scene.path.read_text(encoding="utf-8") and (scene.runs / "v002.py").exists()


def test_improvement_is_accepted(scene):
    scene(MOON)
    out = scene(BALL)
    assert not out.startswith("REGRESSED") and "vs v001: -WARN loose piece (moon)" in out
    assert (scene.runs / "v002.py").exists()


def test_diff_ignores_small_changes():
    m = {"warnings": [["kinks midline", "WARN kinks: 12 ..."]], "profile": {"a->b": 1.0}, "drift": {"a": 2.0}, "seating": {"ear": 0.58}}
    m2 = dict(m, warnings=[["kinks midline", "WARN kinks: 15 ..."]], profile={"a->b": 1.02}, seating={"ear": 0.581})
    assert versions.diff(m, m2) == ([], [])
    worse = dict(m, profile={"a->b": -1.8})
    assert versions.diff(m, worse)[1] == ["worst profile dev 1.0->1.8mm"]
