"""Goals (PROTOCOL §5, §6.6): the LLM states what it wants, the engine searches the numbers.
seat() moves a node's seat knobs until it sits on its host, solve() searches one param until a measure hits a target,
fixes() tries catalogue fixes on top of a rejected batch. Also the size/position helpers the session's queries use."""
import copy
import re

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from . import kernel as K
from .engine import BLEND, SHAPES, FcError, Place, _bbox, _call, _did, _eval, _restore, _s, _snap, parse
from .render import normals
from .senses import measure, seating
from .versions import diff

KNOBS = {"ear": (("flare", 0.0, 0.010, 0.004), ("splay", -45.0, 30.0, 0.0))}  # kind: (param, lo, hi, kernel default), least visible first
TARGET = {"ear": 0.57}  # contact when asked for exactly (attached ears measure 55-60%)
ATTACHED_MIN = {"ear": 0.45}  # seat() without contact=: just clear the 40% warning, changing the look least
ATTACHED = 0.10         # "just attached": this much of a shape's surface touches the rest
TOL = 0.02              # contact / seating tolerance
LEN_TOL = 0.0002        # solve(): length tolerance (m)
PAD = 0.004             # contact(): grid box margin around a part (> the 2 mm surface band)
AXES = ("width", "height", "depth")
BUDGET, SHOW = 4, 2     # fix search: candidate evaluations per reverted batch, fixes shown


def _build(scene, skip=None):
    mod = scene.build(skip)
    if mod is None:
        raise FcError("the scene is empty" if skip is None else f"nothing but {skip} to sit on", "add a body first")
    return mod


def _left(f):
    return lambda p: np.maximum(f(p), -p[:, 0])


def _extent(name, mod, n=64):
    """(lo, hi) of a node's own shape (the left one if mirrored): a grid, then each side refined to the surface."""
    if name not in mod.NODES:
        raise FcError(f'no shape or landmark "{name}"', _did(name, [*mod.NODES, *mod.LANDMARKS], "name"))
    kind, f, mirrored = mod.NODES[name]
    g = _left(f) if mirrored else f
    lo, hi = _bbox(g)
    s = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    P = np.stack(np.meshgrid(*s, indexing="ij"), -1).reshape(-1, 3)
    d = _eval(g, P)
    P, d = P[d < 0], d[d < 0]
    if not len(P):
        return lo, hi
    step, out = (hi - lo) / (n - 1), [P.min(0), P.max(0)]
    for i in range(3):  # ponytail: refined along the axis from the deepest point of the outermost grid layer; ~um on smooth shapes
        for side in (0, 1):
            layer = np.flatnonzero(P[:, i] == out[side][i])
            j = layer[d[layer].argmin()]
            e = np.eye(3)[i] * (2 * side - 1)
            along = lambda t: g((P[j] + t * e)[None])[0]
            if along(step[i]) > 0:
                out[side][i] = P[j, i] + (2 * side - 1) * brentq(along, 0, step[i], xtol=1e-6)
    return out[0], out[1]


def _point(name, mod):
    if name in mod.LANDMARKS:
        return np.asarray(mod.LANDMARKS[name], float)
    lo, hi = _extent(name, mod)
    return (lo + hi) / 2


def contact(part, host, left):
    """Fraction of part's surface within 1.5 mm of host (senses.seating, on its grid but only around the part)."""
    lo, hi = _bbox(_left(part) if left else part)
    return seating(part, host, box=(lo - PAD, hi + PAD), left=left)


def _ear_host(mod, part):
    return next(h for _, p, h in K.ATTACHED if p is part)


def node_contact(scene, mod, name):
    """Contact of a built node with the rest of the model (for an ear: with the head it was attached to)."""
    if name not in mod.NODES:
        raise FcError(f'no shape or ear "{name}"', _did(name, mod.NODES, "node"))
    kind, f, mirrored = mod.NODES[name]
    if kind == "ear":
        return contact(f, _ear_host(mod, f), True)
    if kind not in SHAPES:
        raise FcError(f"a {kind} has no contact measure", "seating works on shapes and ears")
    snap = _snap()
    try:
        return contact(f, _build(scene, skip=name).sdf, mirrored)
    finally:
        _restore(snap)


# ---------- seat ----------

def seat(scene, name, contact=None):
    """Search the node's seat knobs until it sits on its host. Changes the scene; returns the reply as a function of
    the built (model, metrics), so it reports the contact the loop's build really has."""
    kind, P = scene._get(name)
    if kind == "eye":
        raise FcError("an eye has no seating measure yet", f'set its depth with set("{name}", recess=...) and check with look(at="{name}")')
    if kind not in SHAPES and kind != "ear":
        raise FcError(f"can't seat a {kind}", "seat works on shapes and ears")
    if contact is not None and not (type(contact) in (int, float) and 0 < contact < 1):
        raise FcError(f"contact={contact!r}", "a fraction of the surface, e.g. contact=0.57")
    snap = _snap()  # the search builds; the loop's kernel registries must stay as they were
    try:
        return (_seat_ear if kind == "ear" else _seat_shape)(scene, name, contact)
    finally:
        _restore(snap)


def _seat_ear(scene, name, target):
    """Bounded 1-D searches on |contact - target|: flare first, splay only if flare can't reach it; each in widening
    windows around the current value, so the ear keeps the angle the LLM chose whenever possible."""
    floor = target is None  # default: reach at least ATTACHED_MIN; an explicit contact= is hit exactly
    target = ATTACHED_MIN["ear"] if floor else target
    err = (lambda v: max(0.0, target - v)) if floor else (lambda v: abs(v - target))
    ok = 0.0 if floor else TOL
    P = scene.nodes[name][1]
    mod = _build(scene)
    host = _ear_host(mod, mod.NODES[name][1])
    seen = {}

    def c(x):  # one trial: the kernel ear on the same host, contact only
        key = tuple(round(v, 7) for v in x.values())
        if key not in seen:
            K.ear(host, P["tragion"], flare=x["flare"], size=P.get("size", 1.0), splay=x["splay"])
            seen[key] = contact(K.ATTACHED.pop()[1], host, True)
        return seen[key]

    x = {k: float(P.get(k, d)) for k, lo, hi, d in KNOBS["ear"]}
    x0, best = dict(x), c(x)
    # keep the LLM's intent: the knob that changes the look least goes first (KNOBS order), and each search starts in
    # a small window around the current value and only widens if the target isn't reachable there
    for k, lo, hi, _d in KNOBS["ear"]:
        for frac in (0.1, 0.25, 0.5, 1.0):
            if err(best) <= ok:
                break
            w = frac * (hi - lo)
            a, b = max(lo, x[k] - w), min(hi, x[k] + w)
            r = minimize_scalar(lambda v: err(c({**x, k: v})), bounds=(a, b), method="bounded",
                                options=dict(xatol=(hi - lo) / 300, maxiter=12))
            if err(c({**x, k: r.x})) < err(best):
                x[k], best = r.x, c({**x, k: r.x})
    knobs = f"splay={x['splay']:.0f}, flare={x['flare'] * 1000:.1f}mm"
    if err(best) > ok:
        raise FcError(f"seat {name}: contact {target:.0%} not reachable (closest {best:.0%} at {knobs})",
                      f'move the ear (set("{name}", tragion=...)) or ask for less: seat("{name}", contact=...)')
    new = {"splay": round(float(x["splay"]), 1), "flare": round(float(x["flare"]), 4)}
    scene.nodes[name] = ("ear", {**P, **new})
    what = ", ".join(f"splay {x0['splay']:g}->{new['splay']:g}" if k == "splay" else f"flare {x0['flare'] * 1000:.1f}->{new['flare'] * 1000:.1f}mm"
                     for k in new if new[k] != round(x0[k], 4)) or "already seated"
    return lambda mod, m: f"ok seat {name}: {what}, contact {node_contact(scene, mod, name):.0%}"


def _shift(v, d):
    if isinstance(v, Place) and v.fn == "rel":
        return Place("rel", {"name": v.p["name"], **{k: round(float(v.p[k] + x), 5) for k, x in zip(("dx", "dy", "dz"), d)}})
    if isinstance(v, str):
        return Place("rel", {"name": v, **{k: round(float(x), 5) for k, x in zip(("dx", "dy", "dz"), d)}})
    if isinstance(v, Place):
        raise FcError("seat can't move an on() end", "place the ends with rel(...) or (x, y, z)")
    return tuple(round(float(a + x), 5) for a, x in zip(v, d))


def _seat_shape(scene, name, target):
    """Move the shape along the direction to the nearest surface of the rest (its on() depth, if it has one) until
    `target` of its surface touches: the smallest move (brentq on contact - target)."""
    target = ATTACHED if target is None else target
    kind, P = scene.nodes[name]
    if P.get("op", "union") not in ("union", "separate"):
        raise FcError(f"{name} is a {P['op']}", "seat works on added shapes (op union or separate)")
    mod, rest = _build(scene), _build(scene, skip=name)
    part, host, left = mod.NODES[name][1], rest.sdf, bool(P.get("mirror"))

    def reply(what):
        return lambda mod, m: f"ok seat {name}: {what}, contact {contact(mod.NODES[name][1], host, left):.0%}"
    if contact(part, host, left) >= target:
        return reply("already seated")
    lo, hi = _bbox(_left(part) if left else part)
    c = (lo + hi) / 2
    u = -normals(host, c[None], 1e-4)[0]  # toward the nearest surface of the rest, and into it
    t_max = max(float(host(c[None])[0]), 0.0) + float((hi - lo).max())
    gap = lambda t: contact(lambda p: part((K.mirror_x(p) if left else p) - t * u), host, left) - target
    if gap(t_max) < 0:
        raise FcError(f"seat {name}: contact {target:.0%} not reachable (closest {gap(t_max) + target:.0%})",
                      f'move it closer with move("{name}", at=...)')
    t = min(brentq(gap, 0.0, t_max, xtol=2e-4) + 4e-4, t_max)  # + a step past the jump: contact is measured on a grid
    at = P.get("at")
    if isinstance(at, Place) and at.fn == "on":  # ponytail: assumes the on() host's normal ~ u (true when it is the body)
        d0 = at.p["depth"]
        scene.nodes[name] = (kind, {**P, "at": Place("on", {**at.p, "depth": round(float(d0 - t), 5)})})
        return reply(f"depth {d0 * 1000:.1f}->{(d0 - t) * 1000:.1f}mm")
    scene.nodes[name] = (kind, {**P, **{k: _shift(P[k], t * u) for k in ("at", "a", "b") if k in P}})
    return reply(f"moved {t * 1000:.1f}mm toward the body")


# ---------- solve ----------

def _measure(what):
    """A measure name -> f(scene, model) in meters (or a fraction for seating)."""
    w = what.split() if isinstance(what, str) else []
    if isinstance(what, str) and "->" in what:
        a, b = (x.strip() for x in what.split("->", 1))
        return lambda s, mod: float(np.linalg.norm(_point(a, mod) - _point(b, mod)))
    if len(w) == 2 and w[0] in AXES:
        i = AXES.index(w[0])
        return lambda s, mod: float(np.subtract(*_extent(w[1], mod)[::-1])[i])
    if len(w) == 2 and w[0] == "seating":
        return lambda s, mod: node_contact(s, mod, w[1])
    raise FcError(f"unknown measure {what!r}", 'use "width|height|depth <node>", "seating <node>" or "a->b"')


def solve(scene, param, what, target):
    """Bounded 1-D search on one number (a param or a component, "jaw.size[0]") until the measure hits target."""
    m = re.fullmatch(r"([^.\[\]]+)\.(\w+)(?:\[(\d)\])?", param) if isinstance(param, str) else None
    if not m:
        raise FcError(f"bad param {param!r}", 'write "name.param" or "name.param[i]", e.g. "jaw.size[0]"')
    name, k, i = m.group(1), m.group(2), m.group(3) and int(m.group(3))
    kind, P = scene._get(name)
    knobs = {kk: (lo, hi, d) for kk, lo, hi, d in KNOBS.get(kind, ())}
    if k not in P and k not in knobs:
        raise FcError(f'{name} has no param "{k}" set', _did(k, P, "param"))
    v = P.get(k, knobs.get(k, (0, 0, None))[2])
    x0 = v[i] if i is not None and isinstance(v, (tuple, list)) and i < len(v) else v if i is None else None
    if type(x0) not in (int, float):
        raise FcError(f"{param} is not a number ({_s(v)})", 'solve one number; for a tuple add [i], e.g. "jaw.size[0]"')
    if type(target) not in (int, float):
        raise FcError(f"target={target!r}", "a number: meters for lengths, a fraction for seating")
    get, seat_ = _measure(what), isinstance(what, str) and what.startswith("seating")
    show = (lambda y: f"{y:.0%}") if seat_ else (lambda y: f"{y * 1000:.1f}mm")
    lo, hi = knobs[k][:2] if k in knobs else (x0 / 4, 4 * x0) if x0 > 0 else (x0 - 0.05, x0 + 0.05)
    # ponytail: default bounds x0/4..4*x0 (or +-5 cm around 0 or below); a bounds= arg when that falls short

    def put(s, x):
        P2 = dict(s.nodes[name][1])
        val = float(x)
        if i is not None:
            t = list(P2[k])
            t[i] = val
            val = type(P2[k])(t)
        s.nodes[name] = (kind, {**P2, k: val})
    seen = {}

    def f(x):  # one trial: rebuild a copy and measure only this
        if x not in seen:
            s = copy.deepcopy(scene)
            put(s, x)
            try:
                seen[x] = get(s, _build(s))
            except FcError:
                seen[x] = np.inf  # e.g. an on() that misses at this value
        return seen[x]
    snap = _snap()
    try:
        get(scene, _build(scene))  # bad names fail here, with their own hint
        a, b = f(lo) - target, f(hi) - target
        if np.isfinite(a + b) and a * b <= 0:
            brentq(lambda x: f(x) - target, lo, hi, xtol=abs(hi - lo) * 1e-6)
        else:
            minimize_scalar(lambda x: abs(f(x) - target), bounds=(lo, hi), method="bounded", options=dict(xatol=abs(hi - lo) * 1e-5))
    finally:
        _restore(snap)
    x = min(seen, key=lambda x: abs(seen[x] - target))
    if abs(seen[x] - target) > (TOL if seat_ else LEN_TOL):
        raise FcError(f"solve {param}: {what} {show(target)} not reachable in {lo:.4g}..{hi:.4g} (closest {show(seen[x])} at {x:.4g})",
                      "solve another param, or change the target")
    put(scene, round(x, 6))
    return lambda mod, m: f"ok solve {param} {x0:g}->{round(x, 6):g}: {what} {show(get(scene, mod))}"


# ---------- fix search ----------

def _more(P):
    return round(max(2 * P.get("blend", BLEND), 0.005), 4)


def _candidates(scene, key):
    """The catalogue (PROTOCOL §6.6): fix commands for one warning key, most likely first."""
    nodes = scene.nodes
    if key.startswith("loose piece ("):
        n = key[13:-1]
        kind, P = nodes.get(n, ("", {}))
        if kind in SHAPES:
            return [f'seat("{n}")'] + ([f'set("{n}", blend={_more(P)})'] if P.get("op", "union") == "union" else [])
    if key.startswith("seating "):  # kernel parts register as "ear": the engine's ear nodes
        return [f'seat("{n}")' for n, (kind, _) in nodes.items() if kind == "ear"]
    if key.startswith("kinks midline"):  # mirror seam: a larger blend of a mirrored shape (the one it is on first)
        on = key.split(" on ")[-1]
        mir = sorted((n for n, (kind, P) in nodes.items() if kind in SHAPES and P.get("mirror")), key=lambda n: n != on)
        return [f'set("{n}", blend={_more(nodes[n][1])})' for n in mir]
    # kinks on a front_outline trim -> larger k: front_outline isn't an engine node kind yet, so nothing to offer
    if key.startswith("scale: "):
        lint, what = key[7:].split()[:2]
        if lint in K.TYPICAL:
            mid = round(sum(K.TYPICAL[lint][1:]) / 2, 4)
            if nodes.get(lint, ("",))[0] == "stroke" and what == "width" and nodes[lint][1]["brush"] != "carve":
                return [f'set("{lint}", width={mid})']
            if lint == "ear" and what == "length":  # kernel ear length = 62 mm x size
                return [f'set("{n}", size={round(mid / 0.062, 3)})' for n, (kind, _) in nodes.items() if kind == "ear"]
    return []


def fixes(scene, m, bad):
    """For each new warning in `bad`, try catalogue fixes on top of the rejected batch (`scene`, its metrics `m`), so
    the LLM's intent is kept. Keep those that clear the warning and add no new regression. At most BUDGET builds,
    SHOW fixes. -> [(command, predicted result, its reply, the scene's .fc with the fix)]"""
    out, tried = [], 0
    snap = _snap()
    try:
        for key in (b[6:] for b in bad if b.startswith("+WARN ")):
            for cmd in _candidates(scene, key):
                if tried == BUDGET or len(out) == SHOW:
                    return out
                tried += 1
                s = copy.deepcopy(scene)
                c, args, kw = parse(cmd)
                try:
                    r = seat(s, *args, **kw) if c == "seat" else _call(s, c, args, kw)
                    mod = _build(s)
                    mc = measure(mod)
                    r = r(mod, mc) if callable(r) else r
                except FcError:
                    continue
                if key in {k for k, _ in mc["warnings"]} or diff(m, mc)[1]:
                    continue
                what = (r.rsplit(", ", 1)[-1] if c == "seat" else f"seating {mc['seating'].get(key[8:], 0):.0%}"
                        if key.startswith("seating ") else f"no {key}")
                out.append((cmd, f"{what}, warnings: {len(mc['warnings']) or 'none'}", r, s.fc()))
    finally:
        _restore(snap)
    return out


def changed_region(a, ma, b, mb):
    """Boxes around the nodes that differ between scenes a and b (built: ma, mb), before and after, plus a blend
    margin. None = everywhere: a changed node without a shape of its own (mark, curve, stroke) reshapes the skin."""
    names = [n for n in {*a.nodes, *b.nodes} if repr(a.nodes.get(n)) != repr(b.nodes.get(n))]
    boxes = []
    for s, mod in ((a, ma), (b, mb)):
        for n in names:
            if n in s.nodes:
                if mod is None or n not in mod.NODES:
                    return None
                k = s.nodes[n][1].get("blend", BLEND) + 0.005
                lo, hi = _bbox(mod.NODES[n][1])
                boxes.append((lo - k, hi + k))
    return boxes
