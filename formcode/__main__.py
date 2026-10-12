"""usage: python -m formcode scene.py out.png [cx,cy,cz,half] [--keep]   (render + report + version; --keep accepts a regression)
       python -m formcode scene.py out.glb [res]                       (mesh export over the scene's FRAME)
       python -m formcode repl                                         (engine commands from stdin, one batch per line)"""
import sys
import time

from .render import export_mesh
from .versions import run

if sys.argv[1:] == ["repl"]:
    from .engine import repl
    repl()
    sys.exit()
args = [a for a in sys.argv[1:] if a != "--keep"]
if len(args) < 2:
    sys.exit(__doc__)
scene, out = args[:2]
if out.endswith(".glb"):
    t0 = time.time()
    print(export_mesh(scene, out, int(args[2]) if len(args) > 2 else 320), f"{time.time() - t0:.1f}s")
    sys.exit()
print(run(scene, out, keep="--keep" in sys.argv, zoom=[float(x) for x in args[2].split(",")] if len(args) > 2 else None))
