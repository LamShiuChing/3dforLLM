"""Versions: every render run is saved as runs/<scene>/vNNN.py (source) + vNNN.json (metrics + run log), diffed against
the previous accepted version, and reverted when it made things worse (a frozen LLM rarely notices on its own)."""
import json
import time
from collections import Counter
from pathlib import Path

from .render import load, render
from .senses import measure, text

WORSE_MM = 0.5  # worst profile dev / landmark drift sum may grow this much before it counts as a regression


def _numbers(m):
    return {"worst profile dev": max(map(abs, m["profile"].values()), default=None),
            "landmark drift sum": sum(map(abs, m["drift"].values())) if m["drift"] else None}


def diff(old, new):
    """(changes, regressions) between two measure() dicts. Warnings compare by key (kind + subject), not numbers;
    a value counts as changed only when its printed form changes (0.1 mm, 1%)."""
    a, b = Counter(k for k, _ in old["warnings"]), Counter(k for k, _ in new["warnings"])
    bad = [f"+WARN {k}" for k in b - a]
    changes = bad + [f"-WARN {k}" for k in a - b]
    for (label, x), y in zip(_numbers(old).items(), _numbers(new).values()):
        if x is not None and y is not None and f"{x:.1f}" != f"{y:.1f}":
            changes.append(f"{label} {x:.1f}->{y:.1f}mm")
            if y > x + WORSE_MM:
                bad.append(changes[-1])
    for nm in sorted(old["seating"].keys() & new["seating"].keys()):
        x, y = old["seating"][nm], new["seating"][nm]
        if f"{x:.0%}" != f"{y:.0%}":
            changes.append(f"seating {nm} {x:.0%}->{y:.0%}")
    return changes, bad


def run(scene, out, keep=False, zoom=None):
    """Render + report, then version the scene. Returns the CLI text. keep=True accepts a regression (and, right after
    a revert with the scene untouched, brings the rejected version back to accept it)."""
    t0 = time.time()
    scene = Path(scene)
    runs = scene.parent / "runs" / scene.stem
    runs.mkdir(parents=True, exist_ok=True)
    src = scene.read_bytes()
    saved = sorted(runs.glob("v*.py"))
    accepted = [p for p in saved if "-" not in p.stem]
    prev = accepted[-1] if accepted else None
    if keep and prev and saved[-1] != prev and src == prev.read_bytes():
        src = saved[-1].read_bytes()
        scene.write_bytes(src)
    render(str(scene), out, **({"center": zoom[:3], "half": zoom[3]} if zoom else {}))
    m = measure(load(scene))
    head, tail, chars = [], [], len(src.decode("utf-8"))
    if prev and src == prev.read_bytes():
        tag = prev.stem
        tail.append(f"{tag}: scene unchanged, no new version")
    else:
        tag, status = f"v{int(saved[-1].stem[1:4]) + 1 if saved else 1:03d}", "baseline"
        if prev:
            changes, bad = diff(json.loads(prev.with_suffix(".json").read_text(encoding="utf-8"))["metrics"], m)
            status = "accepted" if not bad else "kept" if keep else "rejected"
            tail.append(f"vs {prev.stem}: " + (" | ".join(changes) or "no measurable change"))
            if status == "rejected":
                tag += "-rejected"
                head = [f"REGRESSED vs {prev.stem} -> scene reverted (kept as {tag}). Re-run with --keep to accept.",
                        "why: " + " | ".join(bad)]
                scene.write_bytes(prev.read_bytes())
        (runs / f"{tag}.py").write_bytes(src)
        log = {"version": tag, "status": status, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "chars": chars,
               "tokens": chars // 4, "seconds": round(time.time() - t0, 1), "warnings": len(m["warnings"]), "metrics": m}
        (runs / f"{tag}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    tail.append(f"run {tag}: ~{chars // 4} tokens, {len(m['warnings'])} warnings, {time.time() - t0:.1f}s")
    return "\n".join(head + [text(m)] + tail)
