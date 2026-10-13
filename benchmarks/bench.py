"""Step D benchmark: open a scene, then time a fixed edit sequence (one batch each) through the engine loop.
    .venv/Scripts/python.exe benchmarks/bench.py [face|small] [--runs N] [--profile out.prof]
Prints per-command seconds (the median over N runs: timings are noisy on a busy machine), the first reply line, and
the median of the edits (looks and open excluded)."""
import cProfile
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

from formcode.session import Engine

HERE = Path(__file__).resolve().parent
SMALL = """new('small')
add('head', 'ellipsoid', at=(0, 0, 0), size=(0.075, 0.1, 0.09))
add('cheek_pad', 'ellipsoid', at=(0.04, -0.03, 0.05), size=(0.018, 0.02, 0.016), blend=0.02, mirror=True)
mark('nasion', on('head', x=0, y=0.01))
mark('chin', on('head', x=0, y=-0.08))
curve('profile', through=['nasion', 'chin'])
stroke('nasolabial', 'crease', along=[on('body', x=0.02, y=-0.03), on('body', x=0.03, y=-0.06)], amount=0.0012, width=0.009, mirror=True)
add('ear', 'ear', tragion=(-0.02, -0.005))
"""
EDITS = {
    "face": ['set("nasolabial", amount=0.0015)',
             'move("cheek_pad", at=(0.041, -0.045, 0.056))',
             'add("bump", "sphere", at=on("body", x=0.045, y=-0.03), size=0.004, blend=0.003)',
             'set("ear", flare=0.016)', "keep()", 'seat("ear")',
             'look(at="ear")', 'look("all")'],
    "small": ['set("nasolabial", amount=0.0015)',
              'move("cheek_pad", at=(0.041, -0.03, 0.052))',
              'add("bump", "sphere", at=on("body", x=0.045, y=-0.01), size=0.004, blend=0.003)',
              'set("ear", flare=0.016)', "keep()", 'seat("ear")',
              'look(at="ear")', 'look("all")'],
}


def run(which, prof=None):
    tmp = Path(tempfile.mkdtemp(prefix="fcbench-"))
    if which == "face":
        shutil.copy(HERE / "face.fc", tmp / "face.fc")
    else:
        (tmp / "small.fc").write_text(SMALL, encoding="utf-8")
    e = Engine(root=tmp)
    rows = []
    pr = cProfile.Profile() if prof else None
    for cmd in [f'open("{which}.fc")'] + EDITS[which]:
        t0 = time.perf_counter()
        if pr:
            pr.enable()
        out = e.do(cmd)
        if pr:
            pr.disable()
        rows.append((cmd, time.perf_counter() - t0, out.splitlines()[0]))
    if pr:
        pr.dump_stats(prof)
    shutil.rmtree(tmp, ignore_errors=True)
    return rows


def main(which="face", runs=1, prof=None):
    all_ = [run(which, prof) for _ in range(runs)]
    rows = [(cmd, statistics.median(r[i][1] for r in all_), first) for i, (cmd, _, first) in enumerate(all_[0])]
    for cmd, s, first in rows:
        print(f"{s:6.2f}s  {cmd:<80.80} {first[:70]}")
    edits = [s for cmd, s, _ in rows if not cmd.startswith(("open", "look"))]
    print(f"median edit {statistics.median(edits):.2f}s | total {sum(s for _, s, _ in rows):.1f}s | {runs} run(s)")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a and not a[0].startswith("-") else "face", int(a[a.index("--runs") + 1]) if "--runs" in a else 1,
         a[a.index("--profile") + 1] if "--profile" in a else None)
