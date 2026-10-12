"""usage: python -m formcode scene.py out.png [cx,cy,cz,half]   (render + text report)
       python -m formcode scene.py out.glb [res]              (mesh export over the scene's FRAME)"""
import sys
import time

from .render import export_mesh, load, render
from .senses import report

if len(sys.argv) < 3:
    sys.exit(__doc__)
scene, out = sys.argv[1], sys.argv[2]
t0 = time.time()
if out.endswith(".glb"):
    print(export_mesh(scene, out, int(sys.argv[3]) if len(sys.argv) > 3 else 320), f"{time.time() - t0:.1f}s")
    sys.exit()
if len(sys.argv) > 3:  # zoom
    *c, h = map(float, sys.argv[3].split(","))
    render(scene, out, center=c, half=h)
else:
    render(scene, out)
print(report(load(scene)))
src = open(scene, encoding="utf-8").read()
print(f"render+report {time.time() - t0:.1f}s, scene {len(src)} chars (~{len(src) // 4} tokens)")
