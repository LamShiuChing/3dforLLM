"""usage: python -m formcode scene.py out.png [cx,cy,cz,half] [--keep]   (render + report + version; --keep accepts a regression)
       python -m formcode scene.py out.glb [res]                       (mesh export over the scene's FRAME)
       python -m formcode repl [--allow-code] [--root DIR]             (engine commands from stdin, one batch per line;
                                                                        --allow-code enables code() and opening .py scenes;
                                                                        file paths stay inside --root, default: cwd)
       python -m formcode mcp [--allow-code] [--root DIR]              (the same engine as a stdio MCP server, one tool;
                                                                        needs pip install formcode[mcp])"""
import sys
import time

from .render import export_mesh
from .versions import run


def main():
    if sys.argv[1:2] in (["repl"], ["mcp"]):
        a = sys.argv[2:]
        kw = dict(allow_code="--allow-code" in a, root=a[a.index("--root") + 1] if "--root" in a[:-1] else None)
        if sys.argv[1] == "repl":
            from .session import repl
            repl(**kw)
        else:
            from .mcp_server import serve
            serve(**kw)
        return
    args = [a for a in sys.argv[1:] if a != "--keep"]
    if len(args) < 2:
        sys.exit(__doc__)
    scene, out = args[:2]
    if out.endswith(".glb"):
        t0 = time.time()
        print(export_mesh(scene, out, int(args[2]) if len(args) > 2 else 320), f"{time.time() - t0:.1f}s")
        return
    print(run(scene, out, keep="--keep" in sys.argv, zoom=[float(x) for x in args[2].split(",")] if len(args) > 2 else None))


if __name__ == "__main__":  # spawn workers (render pool, export slabs) must not re-run the CLI
    main()
