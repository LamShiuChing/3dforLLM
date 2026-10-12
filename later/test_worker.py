import re
import sys
from pathlib import Path

import pytest

from formcode.backend.client import Worker, WorkerError, find_blender

ROOT = Path(__file__).resolve().parents[1]


def test_engine_never_imports_bpy():
    import formcode.backend.client  # noqa: F401
    assert "bpy" not in sys.modules
    offenders = [p for p in (ROOT / "formcode").rglob("*.py")
                 if re.search(r"^\s*(import|from)\s+bpy\b", p.read_text(encoding="utf-8"), re.M)]
    assert offenders == []


@pytest.mark.skipif(not find_blender(), reason="Blender not installed")
def test_worker_exports_cube_glb(tmp_path):
    out = tmp_path / "cube.glb"
    with Worker() as w:
        assert w.call("ping")["blender"]
        res = w.call("export_cube", path=str(out))
        with pytest.raises(WorkerError, match="KeyError"):
            w.call("nope")
        assert w.call("ping")  # worker survives a bad request
    assert (res["v"], res["f"]) == (8, 6)
    assert out.read_bytes()[:4] == b"glTF"
