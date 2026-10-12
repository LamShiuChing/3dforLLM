"""The engine (PROTOCOL.md): a scene of named nodes, edited by commands in Python call syntax that are parsed with
`ast` and never executed. Batches are atomic.

Build is staged and deterministic, like examples/face.py:
  1. shapes and code nodes in order -> base (op: union = smin with `blend`, cut, intersect, separate = hard min)
  2. fit the base through every pinned mark and the samples of every pinned curve (kernel.fit)
  3. sculpt strokes in order (kernel.sculpt)
  4. eye / ear parts in order
A placement on "body" sees the model built before its node's stage (in stage 1: the shapes before it).
"""
import ast
import copy
import difflib
import inspect
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from . import kernel as K
from .render import normals
from .senses import measure

BLEND = 0.01  # default blend radius (m) for union / cut / intersect
FAR = 10.0    # on(): rays start this far out (m)
COMMON = ("op", "blend", "mirror", "rot")
SPEC = {  # kind: (required params, optional params)
    "sphere": (("at", "size"), COMMON),
    "ellipsoid": (("at", "size"), COMMON),
    "box": (("at", "size"), COMMON + ("round",)),
    "capsule": (("a", "b", "r"), COMMON + ("r2",)),
    "torus": (("at", "size"), COMMON),
    "eye": (("center", "r", "en", "ex", "up", "lo"), ("lid", "recess")),
    "ear": (("tragion",), ("flare", "size", "splay")),
    "mark": (("at",), ("pin",)),
    "curve": (("through",), ("pin",)),
    "stroke": (("brush", "along", "amount", "width"), ("mirror",)),
    "code": (("src",), ("op", "blend")),
}
ADDABLE = ("sphere", "ellipsoid", "box", "capsule", "torus", "eye", "ear")
SHAPES = ADDABLE[:5]
OPS = ("union", "cut", "intersect", "separate")
BRUSHES = ("crease", "ridge", "inflate", "carve", "groove")
SIDES = {"front": "xy", "back": "xy", "L": "yz", "R": "yz", "top": "xz"}
PLACE_KEYS = ("at", "a", "b", "through", "along")
META = ("PROFILE", "PARTS", "VIEW_NAMES", "FIT_CORRECTIONS", "EPS", "STEP", "MAX_STEPS", "AO")  # taken over from code nodes
REG = ("ATTACHED", "LINT", "EDGES", "DETAIL", "TRIMS", "TYPICAL")  # kernel registries a build fills


class FcError(Exception):
    def __init__(self, cause, hint="", node=None):
        super().__init__(cause)
        self.cause, self.hint, self.node = cause, hint, node


def _close(x, options):
    m = difflib.get_close_matches(str(x), [str(o) for o in options], 1)
    return f'did you mean "{m[0]}"?' if m else ""


def _did(x, options, what):
    opts = list(options)
    return _close(x, opts) or f"{what}s: " + (", ".join(map(str, opts[:12])) + (", ..." if len(opts) > 12 else "") or "none")


def _s(v):
    return v if isinstance(v, str) else repr(v)


# ---------- placement helpers: parsed into Place data, never called ----------

def on(host, x=None, y=None, z=None, side="front", depth=0.0):
    """Cast a ray onto host's skin ("body" or a node name): from the front at (x, y), from side L/R at (y, z),
    from the top at (x, z), from the back at (x, y). depth moves along the surface normal (negative = inside)."""


def rel(name, dx=0.0, dy=0.0, dz=0.0):
    """Offset from a landmark."""


HELPERS = {"on": on, "rel": rel}


class Place:
    def __init__(self, fn, p):
        self.fn, self.p = fn, p

    def __repr__(self):
        (first, v0), *rest = self.p.items()
        if self.fn == "rel":
            return f"rel({', '.join(repr(v) for v in self.p.values())})"
        sig = inspect.signature(on).parameters
        return f"on({', '.join([repr(v0)] + [f'{k}={v!r}' for k, v in rest if v != sig[k].default])})"


# ---------- parser ----------

def _lit(n, helpers=True):
    if isinstance(n, ast.Constant) and type(n.value) is float and not math.isfinite(n.value):
        raise FcError(f"not a finite number: {ast.unparse(n)}", "use a size in meters, e.g. 0.01")
    if isinstance(n, ast.Constant) and (n.value is None or type(n.value) in (int, float, str, bool)):
        return n.value
    if (isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)) and isinstance(n.operand, ast.Constant)
            and type(n.operand.value) in (int, float)):
        return -n.operand.value if isinstance(n.op, ast.USub) else n.operand.value
    if isinstance(n, (ast.Tuple, ast.List)):
        v = [_lit(e, helpers) for e in n.elts]
        return tuple(v) if isinstance(n, ast.Tuple) else v
    if helpers and isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in HELPERS:
        fn, sig = n.func.id, inspect.signature(HELPERS[n.func.id])
        args, kw = _args(n, False)
        try:
            b = sig.bind(*args, **kw)
        except TypeError as e:
            raise FcError(f"{fn}(): {e}", f"usage: {fn}{sig}")
        b.apply_defaults()
        if fn == "on" and b.arguments["side"] not in SIDES:
            raise FcError(f'on(): unknown side "{b.arguments["side"]}"', _did(b.arguments["side"], SIDES, "side"))
        return Place(fn, dict(b.arguments))
    src = ast.unparse(n)
    raise FcError(f"not a literal: {src if len(src) < 30 else src[:27] + '...'}",
                  "use numbers, strings, tuples, lists, True/False, on(...) or rel(...); nothing is evaluated")


def _args(c, helpers):
    if any(k.arg is None for k in c.keywords):
        raise FcError("**kwargs are not allowed", "write each param as name=value")
    return [_lit(a, helpers) for a in c.args], {k.arg: _lit(k.value, helpers) for k in c.keywords}


def parse(line):
    """One command line -> (command, args, kwargs). Literals and on()/rel() only; nothing is executed."""
    try:
        c = ast.parse(line, mode="eval").body
    except SyntaxError as e:
        raise FcError(f"syntax error: {e.msg}", 'one call per line, e.g. add("head", "sphere", at=(0, 0, 0), size=0.1)')
    if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Name)):
        raise FcError("not a command", 'write a call, e.g. tree() or set("ear", splay=-20)')
    if c.func.id not in COMMANDS:
        raise FcError(f'unknown command "{c.func.id}"', _did(c.func.id, COMMANDS, "command"))
    return (c.func.id, *_args(c, True))


# ---------- geometry helpers ----------

@contextmanager
def _tag(name):
    """Errors raised while building a node name that node."""
    try:
        yield
    except FcError as e:
        e.node = e.node or name
        raise
    except Exception as e:
        raise FcError(f"{type(e).__name__}: {e}", "check this node's params", name) from e


def _combine(a, b, op, k):
    if op == "separate" or (op == "union" and k <= 0):
        return lambda p: np.minimum(a(p), b(p))
    k = max(k, 1e-4)
    if op == "union":
        return lambda p: K.smin(a(p), b(p), k)
    if op == "cut":
        return lambda p: K.cut(a(p), b(p), k)
    return lambda p: K.smax(a(p), b(p), k)


def _eval(f, g, n=32768):  # bounded memory: fit() builds (points x 48 x 3) temporaries
    return np.concatenate([f(g[j:j + n]) for j in range(0, len(g), n)])


def _bbox(f):
    """Bounding box of the inside, found by shrinking a grid from a 8 m cube (cells within one cell of the surface
    count as occupied), then one finer pass on the inside itself. Returns (lo, hi) with one cell of margin."""
    c, half = np.zeros(3), 4.0
    for n in (24,) * 6 + (48,):
        s = np.linspace(-half, half, n)
        g = c + np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3)
        h = 2 * half / (n - 1)
        hit = g[_eval(f, g) < (0 if n == 48 else h)]
        if not len(hit):
            if n == 48:
                break  # thinner than the fine grid: keep the coarse box
            raise FcError("the model is empty", "check sizes and op= (a cut can remove everything)")
        lo, hi = hit.min(0) - h, hit.max(0) + h
        c, half = (lo + hi) / 2, (hi - lo).max() / 2
    return lo, hi


def _step(P):
    """Curve sampling for the fit: ~3 samples per gap between control points."""
    return max(float(np.linalg.norm(np.diff(P, axis=0), axis=1).mean()) / 3, 1e-4)


def _radii(T, size):
    """Fit radius per target: 4x the distance to its nearest neighbour target, clipped to 5-20% of the model size
    (dense curve samples get a tight radius, isolated marks a wide smooth one)."""
    # ponytail: one global heuristic, tuned on nothing yet; step F tunes it against the face
    nn = cKDTree(T).query(T, k=2)[0][:, 1] if len(T) > 1 else np.array([np.inf])
    return np.clip(4 * nn, 0.05 * size, 0.2 * size)


def _on(v, ctx):
    p = v.p
    f = ctx["body"] if p["host"] == "body" else ctx["solo"].get(p["host"])
    if f is None:
        raise FcError(f"{v!r}: " + ("no body is built yet here" if p["host"] == "body" else f'no shape "{p["host"]}" built before this'),
                      "place on a shape built earlier, at a landmark, or at (x, y, z)")
    side = p["side"]
    if any(p[a] is None for a in SIDES[side]):
        raise FcError(f'{v!r}: side="{side}" needs {SIDES[side][0]}= and {SIDES[side][1]}=', f"usage: on{inspect.signature(on)}")
    axis, sign = {"front": (2, -1), "back": (2, 1), "L": (0, -1), "R": (0, 1), "top": (1, -1)}[side]
    o = np.array([p["x"] or 0.0, p["y"] or 0.0, p["z"] or 0.0], float)
    o[axis] = -sign * FAR
    d = np.zeros(3)
    d[axis] = sign
    hit = K.cast(f, o, d, t_max=2 * FAR)[0]
    if abs(f(hit[None])[0]) > 1e-4:
        raise FcError(f"{v!r} missed the surface", "move it onto the shape (compare with the shape's position and size)")
    return hit + p["depth"] * normals(f, hit[None], 1e-4)[0] if p["depth"] else hit


def _ren(v, a, b):
    if isinstance(v, str):
        return b if v == a else v
    if isinstance(v, Place):
        first = next(iter(v.p))
        return Place(v.fn, {**v.p, first: _ren(v.p[first], a, b)})
    if isinstance(v, (list, tuple)):
        return type(v)(_ren(x, a, b) for x in v)
    return v


def _absolute(v):
    if isinstance(v, (list, tuple)):
        if len(v) == 3 and all(type(x) in (int, float) for x in v):
            return 1
        return sum(map(_absolute, v))
    return 0


# ---------- the scene ----------

class Scene:
    """Ordered named nodes: name -> (kind, params). Commands are the c_* methods; each returns a short `ok ...`."""

    def __init__(self, name="untitled"):
        self.name, self.nodes, self.writes = name, {}, []

    def call(self, cmd, args, kw):
        fn = getattr(self, "c_" + cmd)
        try:
            b = inspect.signature(fn).bind(*args, **kw)
        except TypeError as e:
            raise FcError(str(e), f"usage: {cmd}{inspect.signature(fn)}")
        return fn(*b.args, **b.kwargs)

    def _get(self, name):
        if name not in self.nodes:
            raise FcError(f'no node "{name}"', _did(name, self.nodes, "node"))
        return self.nodes[name]

    def _free(self, name):
        if not isinstance(name, str) or not name or "." in name or name == "body":
            raise FcError(f"bad name {name!r}", 'names are strings without dots; "body" is reserved')
        if name in self.nodes:
            raise FcError(f'node "{name}" exists', f'change it with set("{name}", ...) or pick another name')

    def _check(self, kind, P):
        req, opt = SPEC[kind]
        for k in P:
            if k not in req + opt:
                raise FcError(f'{kind} has no param "{k}"', (_close(k, req + opt) + " valid: " + ", ".join(req + opt)).strip())
        missing = [k for k in req if k not in P]
        if missing:
            raise FcError(f"{kind} needs {', '.join(missing)}", f"{kind}: {', '.join(req)} [{', '.join(opt)}]")
        if P.get("op", "union") not in OPS:
            raise FcError(f'unknown op "{P["op"]}"', _did(P["op"], OPS, "op"))
        if kind == "stroke":
            if P["brush"] not in BRUSHES:
                raise FcError(f'unknown brush "{P["brush"]}"', _did(P["brush"], BRUSHES, "brush"))
            if (P["brush"] == "carve") != isinstance(P["amount"], (tuple, list)):
                raise FcError("carve takes amount=(rx, ry, rz), other brushes one number", "e.g. amount=0.001")

    def _new(self, name, kind, P):
        self._free(name)
        self._check(kind, P)
        self.nodes[name] = (kind, P)

    # --- session ---
    def c_new(self, name):
        self.name, self.nodes = name, {}
        return f"ok new {name}"

    def c_open(self, path):
        p = Path(path)
        if not p.is_file():
            raise FcError(f"no file {path}", "paths are relative to the working directory")
        if p.suffix not in (".fc", ".py"):
            raise FcError(f"can't open {p.suffix} files", "open a .fc script or a .py scene")
        text = p.read_text(encoding="utf-8")
        self.name, self.nodes = p.stem, {}
        if p.suffix == ".py":  # a legacy scene becomes one code node (its LANDMARKS, PROFILE, ... come along)
            self.nodes[p.stem] = ("code", {"src": text})
            return f"ok opened {p.stem} (1 node: {p.stem}, code)"
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s and not s.startswith("#"):
                try:
                    self.call(*parse(s))
                except FcError as e:
                    raise FcError(f"{p.name} line {i}: {e.cause}", e.hint)
        return f"ok opened {self.name} ({len(self.nodes)} nodes)"

    def c_save(self, path):
        self.writes.append((path, self.fc()))  # written when the batch commits
        return f"ok saved {path}"

    # --- build ---
    def c_add(self, name, shape, **params):
        if shape not in ADDABLE:
            raise FcError(f'unknown shape "{shape}"', _did(shape, ADDABLE, "shape"))
        self._new(name, shape, params)
        return f"ok add {name} ({shape}{', mirror' if params.get('mirror') else ''})"

    def c_mark(self, name, at, pin=True):
        self._new(name, "mark", dict(at=at, pin=pin))
        return f"ok mark {name}" + ("" if pin else " (not pinned)")

    def c_curve(self, name, through, pin=True):
        if not isinstance(through, (list, tuple)) or len(through) < 2:
            raise FcError("through needs a list of 2+ points", 'e.g. through=["nasion", rel("nasion", 0, -0.02, 0.01)]')
        self._new(name, "curve", dict(through=through, pin=pin))
        return f"ok curve {name}: {len(through)} points"

    def c_stroke(self, name, brush, along, amount, width, mirror=False):
        self._new(name, "stroke", dict(brush=brush, along=along, amount=amount, width=width, mirror=mirror))
        return f"ok stroke {name} ({brush})"

    def c_code(self, name, src, **params):
        self._new(name, "code", dict(src=src, **params))
        return f"ok code {name}"

    # --- edit ---
    def c_set(self, name, *value, **params):
        if isinstance(name, str) and "." in name and len(value) == 1 and not params:
            name, k = name.split(".", 1)
            params = {k: value[0]}
        elif value or not params:
            raise FcError("set takes name=value params", f'set("{name}", size=0.1) or set("{name}.size", 0.1)')
        kind, P = self._get(name)
        new = {**P, **params}
        self._check(kind, new)
        self.nodes[name] = (kind, new)
        return f"ok {name} " + ", ".join(f"{k} {_s(P.get(k, 'default'))}->{_s(v)}" for k, v in params.items())

    def c_move(self, name, at):
        kind, P = self._get(name)
        if "at" not in SPEC[kind][0]:
            raise FcError(f"{kind} has no at=", f'change its placement params with set("{name}", ...)')
        P["at"] = at
        return f"ok move {name} -> {_s(at)}"

    def c_rename(self, a, b):
        self._get(a)
        self._free(b)
        self.nodes = {(b if n == a else n): (k, {pk: _ren(v, a, b) if pk in PLACE_KEYS else v for pk, v in P.items()})
                      for n, (k, P) in self.nodes.items()}
        return f"ok rename {a} -> {b}"

    def c_delete(self, name):
        self._get(name)
        del self.nodes[name]
        return f"ok delete {name}"

    def c_tree(self):
        n = len(self.nodes)
        return "\n".join([f"{self.name}: {n} node{'s' * (n != 1)}"] + [
            f"  {name} {k}" + (" mirror" if P.get("mirror") else "") + ("" if P.get("pin", True) else " unpinned")
            for name, (k, P) in self.nodes.items()])

    # --- state ---
    def fc(self):
        """The .fc script of the current state: one replayable line per node."""
        out = [f"new({self.name!r})"]
        for n, (k, P) in self.nodes.items():
            head = f"add({n!r}, {k!r}" if k in ADDABLE else f"{k}({n!r}"
            out.append(head + "".join(f", {a}={v!r}" for a, v in P.items()) + ")")
        return "\n".join(out) + "\n"

    def absolute_placements(self):
        """How many placements are absolute (x, y, z) (PROTOCOL §4: allowed, but counted)."""
        return sum(_absolute(v) for _, P in self.nodes.values() for k, v in P.items() if k in PLACE_KEYS)

    # --- build ---
    def _lm(self, name, ctx):
        if name in ctx["L"]:
            return np.asarray(ctx["L"][name], float)
        kind, P = self.nodes.get(name, (None, None))
        if kind == "mark":  # not resolved yet: shapes in stage 1 see marks that don't need the body
            return self._place(P["at"], dict(ctx, body=ctx["mb"]))
        if isinstance(name, str) and name.endswith("_R") and self.nodes.get(name[:-2] + "_L", ("",))[0] == "mark":
            return self._lm(name[:-2] + "_L", ctx) * (-1, 1, 1)
        names = list(ctx["L"]) + [n for n, (k, _) in self.nodes.items() if k == "mark"]
        raise FcError(f'no landmark "{name}"', _did(name, names, "landmark"))

    def _place(self, v, ctx):
        if isinstance(v, str):
            return self._lm(v, ctx)
        if isinstance(v, Place):
            if v.fn == "rel":
                return self._lm(v.p["name"], ctx) + (v.p["dx"], v.p["dy"], v.p["dz"])
            return _on(v, ctx)
        if isinstance(v, (tuple, list)) and len(v) == 3 and all(type(x) in (int, float) for x in v):
            return np.array(v, float)
        raise FcError(f"bad placement {_s(v)}", "use a landmark name, rel(...), on(...) or (x, y, z)")

    def _shape(self, kind, P, ctx):
        if kind == "capsule":
            a, b = self._place(P["a"], ctx), self._place(P["b"], ctx)
            c = (a + b) / 2
            g = lambda q: K.capsule(q, a, b, P["r"], P.get("r2"))
        else:
            c, s = self._place(P["at"], ctx), P["size"]
            g = {"sphere": lambda q: K.sphere(q, c, s),
                 "ellipsoid": lambda q: K.ellipsoid(q, c, s),
                 "box": lambda q: K.box(q, c, s, P.get("round", 0.0)),
                 "torus": lambda q: K.torus(q, c, *s)}[kind]
        mirror, r = P.get("mirror"), P.get("rot")

        def f(p):
            q = K.mirror_x(p) if mirror else p
            return g(K.rot(q, r[0], r[1], center=c) if r else q)
        return f

    def _code(self, P, ctx):
        ns = {"__name__": "fc_code"}
        exec(compile(P["src"], "<code>", "exec"), ns)  # the escape hatch (PROTOCOL §5): runs locally, like a .py scene
        if not callable(ns.get("sdf")):
            raise FcError("code defines no sdf(p)", "define def sdf(p): returning (N,) distances")
        ctx["L"].update(ns.get("LANDMARKS", {}))
        ctx["meta"].update({k: ns[k] for k in META if k in ns})
        return ns["sdf"]

    def build(self):
        """The staged build (module doc). Returns a scene-module-like model for senses/render, or None if empty."""
        K.reset()
        ctx = dict(body=None, mb=None, solo={}, L={}, curves={}, meta={})
        items = list(self.nodes.items())
        base = None
        for name, (kind, P) in items:  # 1. shapes + code -> base
            if kind in SHAPES or kind == "code":
                with _tag(name):
                    ctx["body"] = base
                    f = self._code(P, ctx) if kind == "code" else self._shape(kind, P, ctx)
                    f(np.zeros((1, 3)))  # bad params fail here, under this node's name
                    ctx["solo"][name] = f
                    base = f if base is None else _combine(base, f, P.get("op", "union"), P.get("blend", BLEND))
        if base is None:
            return None
        ctx["body"] = ctx["mb"] = base
        pins, T = [], []
        for name, (kind, P) in items:  # 2. marks + curves, then fit
            with _tag(name):
                if kind == "mark":
                    ctx["L"][name] = self._place(P["at"], ctx)
                    if P["pin"]:  # x_L marks mirror to x_R (kernel.with_mirrors), and both are pinned
                        pins += [name, name[:-2] + "_R"] if name.endswith("_L") else [name]
                elif kind == "curve":
                    pts = ctx["curves"][name] = np.array([self._place(v, ctx) for v in P["through"]])
                    if P["pin"]:
                        T += list(K.spline(pts, _step(pts)))
        L = ctx["L"] = K.with_mirrors(ctx["L"])
        f, corr = base, {}
        if pins:
            corr = dict(zip(pins, base(np.array([L[n] for n in pins]))))
            T = [L[n] for n in pins] + T
        if T:
            with _tag("fit"):
                T = np.unique(np.array(T), axis=0)
                lo, hi = _bbox(base)
                f = K.fit(base, T, radius=_radii(T, (hi - lo).max()))[0]
        ctx["body"], ops = f, []
        for name, (kind, P) in items:  # 3. strokes
            if kind == "stroke":
                with _tag(name):
                    a = P["along"]
                    if isinstance(a, str) and a not in ctx["curves"]:
                        raise FcError(f'no curve "{a}"', _did(a, ctx["curves"], "curve"))
                    pts = ctx["curves"][a] if isinstance(a, str) else np.array([self._place(v, ctx) for v in a])
                    ops.append((name, P["brush"], pts, P["amount"], P["width"], P["mirror"]))
        if ops:
            with _tag("strokes"):
                f = K.sculpt(f, ops)
                f(np.zeros((1, 3)))
        for name, (kind, P) in items:  # 4. eye / ear
            if kind in ("eye", "ear"):
                with _tag(name):
                    f = (K.eye if kind == "eye" else K.ear)(f, **P)
                    f(np.zeros((1, 3)))
        with _tag("frame"):
            lo, hi = _bbox(f)  # auto-frame: the measured/rendered cube always holds the whole model
        prof = self.nodes.get("profile", ("", {}))
        prof = list(prof[1]["through"]) if prof[0] == "curve" and all(isinstance(v, str) for v in prof[1]["through"]) else []
        m = dict(ctx["meta"])
        m.update(sdf=f, LANDMARKS=L, PROFILE=prof or m.get("PROFILE", []),
                 PARTS={**m.get("PARTS", {}), **{n: g for n, g in ctx["solo"].items()
                                                 if self.nodes[n][0] in SHAPES and self.nodes[n][1].get("op", "union") in ("union", "separate")}},
                 FIT_CORRECTIONS={**m.get("FIT_CORRECTIONS", {}), **corr},
                 FRAME=(*map(float, (lo + hi) / 2), float(0.55 * (hi - lo).max())))
        return SimpleNamespace(**m)


COMMANDS = {n[2:]: n for n in dir(Scene) if n.startswith("c_")}


def _snap():
    return [copy.copy(getattr(K, r)) for r in REG]


def _restore(snap):
    for r, v in zip(REG, snap):
        x = getattr(K, r)
        x.clear()
        x.update(v) if isinstance(x, (dict, set)) else x.extend(v)


def _err(i, line, e):
    cause = f"{e.node}: {e.cause}" if e.node else e.cause
    return (f"error line {i}: {line if len(line) <= 40 else line[:37] + '...'}: {cause}"
            + (f" ({e.hint})" if e.hint else "") + ". Nothing applied.")


class Engine:
    """do(commands) applies a batch atomically (parse + apply to a copy; any error -> nothing applied), rebuilds when
    the scene changed, and replies: the `ok` results joined by " | ", then the warnings."""

    def __init__(self):
        self.scene, self.model, self.metrics = Scene(), None, None

    def do(self, commands):
        work, out, touched, last = copy.deepcopy(self.scene), [], {}, 0
        lines = commands.splitlines()
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            last = i
            try:
                cmd, args, kw = parse(s)
                out.append(work.call(cmd, args, kw))
            except FcError as e:
                return _err(i, s, e)
            except Exception as e:
                return _err(i, s, FcError(f"{type(e).__name__}: {e}"))
            for a in args[:2] if cmd == "rename" else args[:1]:
                if isinstance(a, str):
                    touched[a] = i
        changed, model = work.fc() != self.scene.fc(), self.model
        if changed:
            snap = _snap()
            try:
                model = work.build()
            except FcError as e:
                _restore(snap)  # the kernel registries belong to the model still in use
                i = touched.get(e.node, last)
                return _err(i, lines[i - 1].strip(), e)
        for path, text in work.writes:
            Path(path).write_text(text, encoding="utf-8", newline="\n")
        work.writes = []
        self.scene, self.model = work, model
        reply = " | ".join(out)
        if changed and model is not None:
            self.metrics = measure(model)
            reply += "\n" + ("\n".join(t for _, t in self.metrics["warnings"]) or "warnings: none")
        return reply


def load(path):
    """A .fc script -> its built model (render.load uses this for .fc paths)."""
    s = Scene()
    s.c_open(str(path))
    return s.build()


def repl(lines=None, out=None):
    """One batch per line; a line ending in \\ continues the batch on the next line."""
    eng, buf = Engine(), []
    for line in lines or sys.stdin:
        line = line.rstrip("\r\n")
        if line.endswith("\\"):
            buf.append(line[:-1])
            continue
        text, buf = "\n".join(buf + [line]), []
        if text.strip():
            print(eng.do(text), file=out or sys.stdout, flush=True)
