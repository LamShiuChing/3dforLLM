import json
from pathlib import Path

import numpy as np
import pytest

from formcode.engine import load
from formcode.session import Engine

ROOT = Path(__file__).resolve().parents[1]
BALL = 'add("ball", "sphere", at=(0, 0, 0), size=0.05)'
MOON = 'add("moon", "sphere", at=(0.3, 0, 0), size=0.02)'  # far from the ball: a loose piece


@pytest.fixture
def e():
    e = Engine()
    e.do(BALL)
    return e


def test_accepted_batch_makes_a_version_and_queries_dont(e):
    out = e.do('add("nose", "sphere", at=on("ball", x=0, y=0), size=0.01)').splitlines()
    assert out[0] == "ok add nose (sphere)" and out[-1].startswith("warnings: none | v002 | ")
    assert list(e.h.v) == ["v001", "v002"] and e.h.cur == "v002"
    assert e.do("tree()") == "untitled: 2 nodes\n  ball sphere\n  nose sphere"
    assert e.do("report()").startswith("warnings: none\n--\nALL:")
    assert list(e.h.v) == ["v001", "v002"]


def test_regression_is_reverted_then_kept(e):
    out = e.do(MOON).splitlines()
    assert out[0] == "REVERTED v002: +WARN loose piece (moon)" and out[1].startswith("WARN loose piece (moon)")
    assert out[-1] == "keep() to accept the change anyway"
    assert "moon" not in e.do("tree()") and e.h.v["v002-rejected"]["status"] == "rejected" and e.h.cur == "v001"
    out = e.do("keep()").splitlines()
    assert out[0] == "ok keep v002-rejected (was: +WARN loose piece (moon))" and out[1] == "+WARN loose piece (moon)"
    assert out[2].startswith("1 warning | v003")
    assert "moon" in e.do("tree()") and "v003 kept, 1 warn" in e.do("history()")
    assert e.do("keep()").startswith("error line 1: keep(): the scene changed since v002-rejected")


def test_undo_and_goto(e):
    e.do('add("nose", "sphere", at=on("ball", x=0, y=0), size=0.01)')
    out = e.do("undo()").splitlines()
    assert out[0] == "ok undo -> v001" and out[-1].startswith("warnings: none | v003")
    assert "nose" not in e.do("tree()")
    assert e.do("undo()").startswith("error line 1: undo(): can't undo 1")  # v003 is v001 again: nothing before it
    assert e.do('goto("v002")').startswith("ok goto -> v002") and "nose" in e.do("tree()")
    out = e.do('goto("v02")')
    assert 'no version "v02"' in out and "did you mean" in out and "Nothing applied" in out


def test_versions_persist_once_the_scene_has_a_path(e, tmp_path):
    e.do(f'save("{(tmp_path / "s.fc").as_posix()}")')
    e.do('add("nose", "sphere", at=on("ball", x=0, y=0), size=0.01)')
    e.do(MOON)
    runs = tmp_path / "runs" / "s"
    assert sorted(p.name for p in runs.iterdir()) == ["v001.fc", "v001.json", "v002.fc", "v002.json",
                                                       "v003-rejected.fc", "v003-rejected.json"]
    log = json.loads((runs / "v002.json").read_text(encoding="utf-8"))
    assert log["status"] == "accepted" and log["tokens"] > 0 and log["warnings"] == 0 and "metrics" in log
    assert json.loads((runs / "v003-rejected.json").read_text(encoding="utf-8"))["status"] == "rejected"
    assert "nose" in (runs / "v002.fc").read_text(encoding="utf-8")
    e2 = Engine()  # reopening continues the numbering: nothing is overwritten
    assert e2.do(f'open("{(tmp_path / "s.fc").as_posix()}")').splitlines()[-1].startswith("warnings: none | v004")


def test_reply_length_is_capped():
    batch = "\n".join(f'add("a_rather_long_part_name_{i:02d}", "sphere", at=({i * 0.01}, 0, 0), size=0.02)' for i in range(20))
    out = Engine().do(batch)
    assert len(out) <= 600 and "more)" in out.splitlines()[0] and "v001" in out


def test_measure_between_marks_and_node_size(e):
    e.do('mark("a", on("ball", x=0, y=0))\nmark("b", on("ball", y=0, z=0, side="L"))')
    assert e.do('measure("a", "b")') == f"a->b {np.hypot(0.05, 0.05) * 1000:.1f}mm"
    w = e.do('measure("ball")')
    assert w.startswith("ball: ") and abs(float(w.split()[1].split("x")[0]) - 100) < 3
    assert "did you mean" in e.do('measure("bal", "a")')


def test_look_writes_an_image(e, tmp_path):
    e.do(f'save("{(tmp_path / "s.fc").as_posix()}")\nadd("nose", "sphere", at=on("ball", x=0, y=0), size=0.01)')
    out = e.do('look(view="front", at="nose", size=32, before=True)')
    path = out.split()[1]
    assert out.startswith("[image] ") and "front at nose, before|now" in out and Path(path).is_file()
    assert Path(path).parent == tmp_path / "runs" / "s" and len(e.h.v) == 2
    assert "unknown view" in e.do('look(view="left")')


def test_export_glb_and_sidecar(e, tmp_path):
    pytest.importorskip("trimesh")
    e.do('mark("tip", on("ball", x=0, y=0))\ncurve("c", through=["tip", on("ball", x=0.02, y=0)])')
    out = e.do(f'export("{(tmp_path / "b.glb").as_posix()}", res=40)')
    assert out.startswith("ok exported b.glb: ") and "watertight=True" in out
    side = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert "tip" in side["landmarks"] and len(side["curves"]["c"]) > 2 and side["nodes"][0] == {"name": "ball", "kind": "sphere"}


def test_code_is_opt_in(tmp_path):
    face = (ROOT / "examples" / "face.py").as_posix()
    script = tmp_path / "c.fc"
    script.write_text('code("blob", "from formcode import *\\nsdf = lambda p: sphere(p, (0, 0, 0), 0.1)")\n', encoding="utf-8")
    e = Engine()
    for batch in ['code("x", "def sdf(p): return p[:, 0]")', f'open("{face}")', f'open("{script.as_posix()}")']:
        out = e.do(batch)
        assert "code execution is off" in out and "--allow-code" in out and "Nothing applied" in out
    assert not e.scene.nodes and not e.h.v
    out = Engine(allow_code=True).do(f'open("{script.as_posix()}")')
    assert out.startswith("ok opened c (1 nodes)") and "v001" in out
    assert load(script).sdf(np.zeros((1, 3)))[0] < 0  # a file the user runs on purpose (CLI, render workers)
