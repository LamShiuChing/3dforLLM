"""The always-on session (PROTOCOL §6-§8): after every batch that changes the scene, rebuild, run every sense, diff
against the last accepted version, then save a version or revert a regression, and reply in the §7 format.
Also the commands that act on the session rather than the scene: history (undo, goto, keep, history) and queries
(report, measure, look, export). The scene, the command language and the build are in engine.py."""
import copy
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import goals
from .engine import COMMANDS, FcError, Scene, _call, _did, _err, _restore, _snap, inside, parse
from .goals import _extent, _point
from .gridcache import GridCache
from .render import DEFAULT_VIEWS, compare, frame, render
from .senses import measure, mm, text
from .versions import History, verdict

LOOK = {"front": ["front"], "side": ["side (from +x)"], "3/4": ["3/4"], "back": ["back"], "top": ["top"],
        "all": DEFAULT_VIEWS + ["back"]}
CAP = 600                                   # reply chars (~150 tokens, PROTOCOL §7), except the listings in LONG
LONG = ("tree", "history", "report")
QUERIES = LONG + ("measure", "look", "export")  # these see the batch's result (run after the build, before the verdict)


def _runs(path):
    return Path(path).parent / "runs" / Path(path).stem if path else None


def _cut(s, n):
    return s if len(s) <= n else s[:n - 3] + "..."


def _cap(items, n, more=""):
    """items joined by " | " in at most n chars; whole items only, the rest counted."""
    s = " | ".join(items)
    if len(s) <= n:
        return s
    s = s[:n - 30 - len(more)]
    s = s.rsplit(" | ", 1)[0] if " | " in s else s
    return s + f" ... (+{len(items) - s.count(' | ') - 1} more{more})"


def _warnings(m, old, tail):
    """Line 3 of a reply: the count, version and time, then up to 3 warnings, new ones first."""
    W = m["warnings"] if m else []
    if not W:
        return [f"warnings: none | {tail}"]
    seen = {k for k, _ in old["warnings"]} if old else set()
    W = sorted(W, key=lambda w: w[0] in seen)
    more = f" (+{len(W) - 3} more: report())" if len(W) > 3 else ""
    return [f"{len(W)} warning{'s' * (len(W) > 1)} | {tail}{more}"] + [_cut(t, 110) for _, t in W[:3]]


def _need(x):
    if x is None:
        raise FcError("the scene is empty", "add a shape first")
    return x


class Engine:
    """do(commands) applies a batch atomically (parse + apply to a copy; any error -> nothing applied). When the scene
    changed it rebuilds, measures, diffs against the last accepted version and saves a version, or reverts a
    regression (PROTOCOL §6); replies per §7. History and query commands are the c_* methods here."""

    def __init__(self, allow_code=False, root=None):
        self.root = Path(root or Path.cwd()).resolve()  # the workspace: open/save/export paths must stay inside it
        self.scene, self.model, self.metrics = Scene(allow_code=allow_code, root=self.root), None, None
        self.h, self.b, self.tmp, self.looks = History(), None, None, 0
        self.cache = GridCache()  # the measure grid per build stage: an edit re-evaluates only what it changed

    def do(self, commands):
        t0 = time.time()
        work, out, touched, last = copy.deepcopy(self.scene), [], {}, 0
        self.b = b = SimpleNamespace(work=work, base=None, fresh=False, prev=None)  # what history commands act on
        lines = commands.splitlines()
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            last = i
            try:
                cmd, args, kw = parse(s)
                out.append((i, s, cmd, _call(self if hasattr(Engine, "c_" + cmd) else work, cmd, args, kw)))
            except Exception as e:
                return _err(i, s, e if isinstance(e, FcError) else FcError(f"{type(e).__name__}: {e}"))
            if cmd in ("new", "open"):
                b.fresh, b.base = True, None
            for a in args[:2] if cmd == "rename" else args[:1]:
                if isinstance(a, str):
                    touched[a] = i
        fc = work.fc()
        changed, model, m, snap = b.fresh or fc != self.scene.fc(), self.model, self.metrics, _snap()
        h = History(_runs(work.path)) if b.fresh else self.h
        if changed:
            try:
                model = work.build()
            except FcError as e:
                _restore(snap)  # the kernel registries belong to the model still in use
                i = touched.get(e.node, last)
                return _err(i, lines[i - 1].strip(), e)
            m = measure(model, cache=self.cache) if model is not None else None
        cur = h.get(h.cur)
        b.prev = cur if changed else cur and h.get(cur["parent"])  # what look(before=True) compares with
        res = []
        for i, s, cmd, r in out:  # queries run on the batch's result
            try:
                res.append((cmd, r(model, m) if callable(r) else r))
            except Exception as e:
                _restore(snap)
                return _err(i, s, e if isinstance(e, FcError) else FcError(f"{type(e).__name__}: {e}"))
        ver = ref = None
        if changed:
            same = b.base is not None and fc == b.base["fc"]  # undo / goto / keep: no regression check
            ref = cur if same or b.base is None else b.base
            status, changes, bad = verdict(ref and ref["metrics"], m)
            if any(x.startswith("+WARN kinks") for x in bad):  # a new kink cluster counts only near the change
                status, changes, bad = verdict(ref["metrics"], m, region=goals.changed_region(self.scene, self.model, work, model))
            if same:
                status, bad = ("kept" if b.base["status"] == "rejected" else "accepted"), []
            parent = (b.base["parent"] if same else b.base["version"]) if b.base else h.cur
            if bad:
                _restore(snap)
                e = h.add(fc, commands, t0, m, "rejected", parent, changes, why=bad)
                return self._reverted(e, m, res, t0)
        if work.path != self.scene.path and not b.fresh:
            h.move(_runs(work.path))
        if changed:
            ver = h.add(fc, commands, t0, m, status, parent, changes)
        for path, t in work.writes:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(t, encoding="utf-8", newline="\n")
        work.writes = []
        self.scene, self.model, self.metrics, self.h = work, model, m, h
        rest = []
        if ver:
            rest = ([_cap(ver["changes"], 150)] if ver["changes"] else []) + _warnings(
                m, ref and ref["metrics"], f"{ver['version']} | {time.time() - t0:.1f}s")
        short = [r for cmd, r in res if cmd not in LONG]
        head = [_cap(short, max(80, CAP - len("\n".join(rest)) - 1))] if short else []
        return "\n".join(head + rest + [r for cmd, r in res if cmd in LONG])

    def _reverted(self, e, m, res, t0):
        bad = e["why"]
        fx = e["fixes"] = goals.fixes(self.b.work, m, bad, self.cache)
        out = [_cut(f"REVERTED {e['version'].split('-')[0]}: " + " | ".join(bad), 200)]
        out += [_cut(t, 110) for k, t in m["warnings"] if f"+WARN {k}" in bad][:3]
        out += [f"fix {j}: {c} -> {r}" for j, (c, r, *_) in enumerate(fx, 1)]
        out += [f"back at {self.h.cur} | {time.time() - t0:.1f}s",
                ("fix(1) to apply | " if fx else "") + "keep() to accept the change anyway"]
        return "\n".join(out + [r for cmd, r in res if cmd in QUERIES])

    # --- history (PROTOCOL §5): undo / goto / keep load a saved version into the batch's scene ---
    def _to(self, e):
        w = self.b.work
        path = w.path
        w.replay(e["fc"], e["version"])
        w.path, self.b.base = path, e
        return e["version"]

    def c_undo(self, n=1):
        if type(n) is not int or n < 1:
            raise FcError(f"undo({n!r})", "n counts versions back, e.g. undo() or undo(2)")
        e = self.b.base or self.h.get(self.h.cur)
        for _ in range(n):
            e = e and self.h.get(e["parent"])
        if e is None:
            raise FcError(f"can't undo {n}: no accepted version that far back", "history() lists the versions")
        return f"ok undo -> {self._to(e)}"

    def c_goto(self, tag):
        e = self.h.get(tag)
        if e is None:
            raise FcError(f'no version "{tag}"', _did(tag, self.h.v, "version"))
        return f"ok goto -> {self._to(e)}"

    def c_keep(self):
        e = self.h.last_rejected()
        if e is None:
            raise FcError("nothing to keep", "keep() accepts the last REVERTED batch")
        if self.scene.fc() != self.h.get(e["parent"])["fc"]:
            raise FcError(f"the scene changed since {e['version']}", f'goto("{e["version"]}") brings it back anyway')
        self._to(e)
        return f"ok keep {e['version']} (was: {' | '.join(e['why'])})"

    def c_fix(self, n=1):
        e = self.h.last_rejected()
        fx = e and e.get("fixes")
        if not fx:
            raise FcError("no fixes to apply", "fixes are offered after a REVERTED batch; keep() accepts one as it is")
        if type(n) is not int or not 1 <= n <= len(fx):
            raise FcError(f"no fix {n!r}", "the last REVERTED batch offered fix(1)" + (f" to fix({len(fx)})" if len(fx) > 1 else ""))
        if self.scene.fc() != self.h.get(e["parent"])["fc"]:
            raise FcError(f"the scene changed since {e['version']}", f'goto("{e["version"]}") brings the rejected batch back')
        w = self.b.work
        path = w.path
        w.replay(fx[n - 1][3], e["version"])  # the rejected batch + the fix, as checked by the fix search
        w.path = path
        return fx[n - 1][2]

    # --- goals (PROTOCOL §5): the engine searches the numbers; the reply comes from the loop's build ---
    def c_seat(self, name, contact=None):
        return goals.seat(self.b.work, name, contact)

    def c_solve(self, param, measure, target):
        return goals.solve(self.b.work, param, measure, target)

    def c_history(self):
        return self.h.lines()

    # --- queries: they return a function of (model, metrics), run once the batch is built ---
    def c_report(self):
        return lambda mod, m: text(_need(m))

    def c_measure(self, a, b=None):
        def q(mod, m):
            _need(mod)
            if b is not None:
                return f"{a}->{b} {np.linalg.norm(_point(a, mod) - _point(b, mod)) * 1000:.1f}mm"
            if a in mod.LANDMARKS:
                d = m["drift"].get(a)
                return f"{a}: at {mm(mod.LANDMARKS[a])}mm" + ("" if d is None else f", {d:+.1f}mm off the surface")
            if a in mod.CURVES:
                return f"{a}: {np.linalg.norm(np.diff(mod.CURVES[a], axis=0), axis=1).sum() * 1000:.0f}mm long"
            if self.b.work.nodes.get(a, ("",))[0] == "stroke":
                raise FcError(f'"{a}" is a stroke: it has no size of its own', "measure(a, b) between marks along it")
            lo, hi = _extent(a, mod)
            kind, _, mirrored = mod.NODES[a]
            seat = m["seating"].get("ear") if kind == "ear" else None
            return (f"{a}{' (left)' if mirrored else ''}: {'x'.join(f'{v * 1000:.0f}' for v in hi - lo)}mm (x y z) "
                    f"at {mm((lo + hi) / 2)}mm" + ("" if seat is None else f", seating {seat:.0%}"))
        return q

    def c_look(self, view="all", at=None, size=384, before=False):
        if view not in LOOK:
            raise FcError(f'unknown view "{view}"', _did(view, LOOK, "view"))
        if type(size) is not int or not 16 <= size <= 2048:
            raise FcError(f"size={size!r}", "a pixel size, e.g. size=384")

        def q(mod, m):
            c, half = frame(_need(mod))
            if at in mod.LANDMARKS:
                c, half = np.asarray(mod.LANDMARKS[at], float), 0.25 * half
            elif at is not None:
                lo, hi = _extent(at, mod)
                c, half = (lo + hi) / 2, max(0.6 * float((hi - lo).max()), 0.005)
            self.looks += 1
            out = self._dir() / f"look{self.looks:03d}-{view.replace('/', '')}.png"
            res = size if len(LOOK[view]) == 1 else size // 2
            render(None, str(out), res, c, half, names=LOOK[view], mod=mod, fc=self.b.work.fc())
            if before:
                p = self.b.prev
                if p is None:
                    raise FcError("no previous version to compare with", "look(before=True) needs two versions")
                snap, s = _snap(), Scene(allow_code=True)
                try:
                    s.replay(p["fc"], p["version"])
                    pm = _need(s.build())
                finally:
                    _restore(snap)
                bo = out.with_name(out.stem + "-before.png")
                render(None, str(bo), res, c, half, names=LOOK[view], mod=pm, fc=p["fc"])
                compare(bo, out, out, (p["version"], "now"))
            return f"[image] {out.as_posix()} ({view}{f' at {at}' if at else ''}{', before|now' if before else ''}, {2 * half:.3g} m across)"
        return q

    def c_export(self, path, res=320):
        p = inside(self.root, path, (".glb",))  # + its .json sidecar

        def q(mod, m):
            _need(mod)
            w = self.b.work
            src = self._dir() / "_export.fc"  # the mesh workers rebuild the scene from a file
            src.write_text(w.fc(), encoding="utf-8", newline="\n")
            try:
                from .render import make_mesh
                mesh = make_mesh(str(src), res)
            except ImportError:
                raise FcError("export needs scikit-image and trimesh", "pip install formcode[mesh]")
            finally:
                src.unlink()
            mesh.export(p)
            side = {"scene": w.name, "frame": list(mod.FRAME), "nodes": [{"name": n, "kind": k} for n, (k, _) in w.nodes.items()],
                    "landmarks": {k: [round(float(x), 6) for x in v] for k, v in mod.LANDMARKS.items()},
                    "curves": {k: np.round(C, 6).tolist() for k, C in mod.CURVES.items()}}
            p.with_suffix(".json").write_text(json.dumps(side, indent=1), encoding="utf-8")
            return f"ok exported {p.name}: {len(mesh.vertices)} verts, watertight={mesh.is_watertight}"
        return q

    def _dir(self):
        """Where look() / export() put files: the scene's runs dir, or a temp dir for a scene without a path."""
        d = _runs(self.b.work.path)
        if d is None:
            self.tmp = d = self.tmp or Path(tempfile.mkdtemp(prefix="formcode-"))
        d.mkdir(parents=True, exist_ok=True)
        return d


COMMANDS.update({n[2:]: n for n in dir(Engine) if n.startswith("c_")})


def repl(lines=None, out=None, allow_code=False, root=None):
    """One batch per line; a line ending in \\ continues the batch on the next line."""
    eng, buf = Engine(allow_code, root), []
    for line in lines or sys.stdin:
        line = line.rstrip("\r\n")
        if line.endswith("\\"):
            buf.append(line[:-1])
            continue
        text, buf = "\n".join(buf + [line]), []
        if text.strip():
            print(eng.do(text), file=out or sys.stdout, flush=True)
