"""Versions: every change is saved as runs/<scene>/vNNN.<ext> (source) + vNNN.json (metrics + run log), diffed against
the previous accepted version, and reverted when it made things worse (a frozen LLM rarely notices on its own).
Two users share the rules here: the file CLI (`run`, scene.py on disk) and the engine (`History`, PROTOCOL §6)."""
import json
import re
import time
from collections import Counter
from pathlib import Path

from .render import load, render
from .senses import measure, text

WORSE_MM = 0.5  # worst profile dev / landmark drift sum may grow this much before it counts as a regression
KINK_MIN = 5    # a new kink cluster needs this many samples to count as a regression


def _counts(key, m, region):
    """Does a new warning count as a regression? Kink clusters only with >= KINK_MIN samples and inside the changed
    region (boxes, None = anywhere): the kink samples are redrawn from the whole shell on every build, so small
    clusters far from an edit come and go (they are still shown as warnings)."""
    c = m.get("kink_clusters", {}).get(key)
    if c is None:
        return True
    lo, hi = c[1:4], c[4:7]
    return c[0] >= KINK_MIN and (region is None or any(
        all(a <= q and b >= p for a, b, p, q in zip(lo, hi, blo, bhi)) for blo, bhi in region))


def _numbers(m):
    return {"worst profile dev": max(map(abs, m["profile"].values()), default=None),
            "landmark drift sum": sum(map(abs, m["drift"].values())) if m["drift"] else None}


def diff(old, new, region=None):
    """(changes, regressions) between two measure() dicts. Warnings compare by key (kind + subject), not numbers;
    a value counts as changed only when its printed form changes (0.1 mm, 1%). region: see _counts."""
    a, b = Counter(k for k, _ in old["warnings"]), Counter(k for k, _ in new["warnings"])
    changes = [f"+WARN {k}" for k in b - a]
    bad = [x for x in changes if _counts(x[6:], new, region)]
    changes += [f"-WARN {k}" for k in a - b]
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


def verdict(old, new, keep=False, region=None):
    """(status, changes, regressions) of new metrics vs the last accepted ones (None = no version yet / empty scene)."""
    if old is None or new is None:
        return ("baseline" if old is None else "accepted"), [], []
    changes, bad = diff(old, new, region)
    return ("accepted" if not bad else "kept" if keep else "rejected"), changes, bad


def next_number(runs, used=0):
    """The next free version number: after every vNNN file in runs (any extension) and after `used`."""
    nums = [int(m.group(1)) for p in (Path(runs).glob("v*") if runs else []) if (m := re.match(r"v(\d+)", p.name))]
    return max(nums + [used]) + 1


def record(runs, tag, status, src, ext, t0, m, **extra):
    """The run log of one version; written to runs/<tag>.json (+ the source as runs/<tag><ext> if ext) when runs is set."""
    log = {"version": tag, "status": status, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "chars": len(src),
           "tokens": len(src) // 4, "seconds": round(time.time() - t0, 1), "warnings": len(m["warnings"]) if m else 0,
           "metrics": m, **extra}
    if runs:
        Path(runs).mkdir(parents=True, exist_ok=True)
        if ext:
            (Path(runs) / f"{tag}{ext}").write_text(src, encoding="utf-8", newline="\n")
        (Path(runs) / f"{tag}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log


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
        old = json.loads(prev.with_suffix(".json").read_text(encoding="utf-8"))["metrics"] if prev else None
        status, changes, bad = verdict(old, m, keep)
        tag = f"v{next_number(runs):03d}"
        if prev:
            tail.append(f"vs {prev.stem}: " + (" | ".join(changes) or "no measurable change"))
        if status == "rejected":
            tag += "-rejected"
            head = [f"REGRESSED vs {prev.stem} -> scene reverted (kept as {tag}). Re-run with --keep to accept.",
                    "why: " + " | ".join(bad)]
            scene.write_bytes(prev.read_bytes())
        (runs / f"{tag}.py").write_bytes(src)  # exact bytes: the unchanged check above compares them
        record(runs, tag, status, src.decode("utf-8"), None, t0, m)
    tail.append(f"run {tag}: ~{chars // 4} tokens, {len(m['warnings'])} warnings, {time.time() - t0:.1f}s")
    return "\n".join(head + [text(m)] + tail)


class History:
    """The engine's versions: in memory (tag -> run log + "fc"), and under `runs` once the scene has a path.
    `cur` is the accepted version the scene is at; each version's `parent` is the one it was made from (undo walks it)."""

    def __init__(self, runs=None):
        self.v, self.runs, self.cur = {}, runs, None

    def get(self, tag):
        return self.v.get(tag)

    def last_rejected(self):
        return next((e for e in reversed(self.v.values()) if e["status"] == "rejected"), None)

    def add(self, fc, batch, t0, m, status, parent, changes, why=()):
        n = next_number(self.runs, max((int(t[1:4]) for t in self.v), default=0))
        tag = f"v{n:03d}" + ("-rejected" if status == "rejected" else "")
        e = record(self.runs, tag, status, fc, ".fc", t0, m, chars=len(batch), tokens=len(batch) // 4,
                   parent=parent, batch=batch, changes=changes, why=list(why))
        self.v[tag] = dict(e, fc=fc)
        if status != "rejected":
            self.cur = tag
        return self.v[tag]

    def move(self, runs):
        """The scene got a path: persist from now on; the versions so far go along only into an empty runs dir."""
        if runs == self.runs:
            return
        if next_number(runs) == 1:
            for e in self.v.values():
                record(runs, e["version"], e["status"], e["fc"], ".fc", time.time(), e["metrics"],
                       **{k: e[k] for k in ("time", "chars", "tokens", "seconds", "parent", "batch", "changes", "why")})
        self.runs = runs

    def lines(self):
        out = []
        for t, e in self.v.items():
            first = (e["changes"] or [e["batch"].strip().splitlines()[0] if e["batch"].strip() else ""])[0]
            out.append(f"{t} {e['status']}, {e['warnings']} warn: {first[:50]}" + (" <- now" if t == self.cur else ""))
        return "\n".join(out) or "no versions yet"
