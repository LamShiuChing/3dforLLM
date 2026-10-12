"""Golden tests: every example's text report and a 64 px render of every view must not drift.
Regenerate after an intended change: python tests/test_examples.py"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from formcode.render import DEFAULT_VIEWS, frame, load, render_view
from formcode.senses import report

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "tests" / "golden"
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))


def views(path):
    mod = load(path)
    c, h = frame(mod)
    names = getattr(mod, "VIEW_NAMES", DEFAULT_VIEWS)
    with ProcessPoolExecutor(len(names)) as ex:
        return dict(ex.map(render_view, [(str(path), n, 64, c, h) for n in names]))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_report(path):
    assert report(load(path)) == (GOLD / f"{path.stem}.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_render(path):
    gold = np.load(GOLD / f"{path.stem}.npz")
    imgs = views(path)
    assert sorted(imgs) == sorted(gold.files)
    for name, img in imgs.items():
        changed = (np.abs(img.astype(int) - gold[name].astype(int)) > 8).mean()
        assert changed < 0.005, f"{name}: {changed:.1%} of pixels changed"


if __name__ == "__main__":
    GOLD.mkdir(exist_ok=True)
    for path in EXAMPLES:
        (GOLD / f"{path.stem}.txt").write_text(report(load(path)), encoding="utf-8", newline="\n")
        np.savez_compressed(GOLD / f"{path.stem}.npz", **views(path))
        print("updated", path.stem)
