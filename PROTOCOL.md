# formcode protocol — v1 (2026-10-12, decisions accepted)

**Idea:** the engine is always on and does the heavy lifting. The LLM only says *what*, *where* and *what it's called*.
After every change the engine rebuilds what changed, runs every check, saves a version, undoes regressions, tests fixes,
and replies with what changed. The LLM never has to remember to check anything.

## 1. What a session looks like
```
LLM ▸ open("face.fc")
eng ◂ ok opened face (12 nodes)
      warnings: none | v001 | 4.1s

LLM ▸ set("ear", splay=-35)
eng ◂ REVERTED v002: +WARN seating ear
      WARN seating ear: only 32% in contact: stuck on / floating (attached ears measure ~55-60%)
      fix 1: seat("ear") -> seating 56%, warnings: none
      fix 2: set("ear", splay=-12) -> seating 48%, warnings: none
      back at v001 | 1.1s
      fix(1) to apply | keep() to accept the change anyway

LLM ▸ fix(1)
eng ◂ ok seat ear: splay -35->-14, flare 4.0->3.1mm
      seating ear 58%->56%
      warnings: none | v003 | 1.2s

LLM ▸ look(at="ear", view="side")
eng ◂ [image] runs/face/look001-side.png (side at ear, 0.06 m across)
```
Four short messages, and one image that the LLM asked for. Every check ran on every change without being asked.

## 2. Principles
1. **Push, don't pull.** Checks, versioning and diffs happen automatically after every change. Queries are for extra detail only.
2. **Intent in, numbers solved.** The LLM states goals (`seat`, `solve`); the engine searches the numbers.
3. **Everything has a name.** Nodes, landmarks and curves are named. Replies, errors and fixes all refer to names.
4. **Relative by default.** Positions come from landmarks, other nodes or the skin. Absolute xyz is allowed, but it is counted and reported.
5. **Short replies.** At most 150 tokens, except `report()` and `look()`. Changes only, never the full state.
6. **Batches are atomic.** If any command in a batch fails, nothing is applied. The checks run once, at the end of the batch.
7. **Deterministic.** The same command sequence always builds the same model.

## 3. Syntax: Python call syntax, never executed
- One command per line, written as a Python function call with literal arguments: numbers, strings, tuples, lists, `True`/`False`, and the placement helpers in §4.
- The engine parses commands with Python's `ast` module and **never executes them**. That makes them safe and deterministic, and every error comes with a precise fix hint.
- The one exception is code: `code()` nodes and opening a legacy `.py` scene run Python. That is **opt-in**: off by default, on only when the engine is started with `--allow-code` (`Engine(allow_code=True)`). With it off, `code(...)`, a `.fc` file containing a `code(...)` line and `open("x.py")` fail with `code execution is off (start the engine with --allow-code ...)` and nothing is applied. Reason: behind MCP, an LLM (or a prompt-injected one, or a downloaded `.fc` file) must not be able to run arbitrary code.
- Units are meters. Axes: y up, +z front, x = the subject's left. `mirror=True` makes the node exist on both sides (define it on the left, x>0).
- Dotted names reach inside a node: `"ear.splay"` is the `splay` parameter of node `ear`.

## 4. Placement: where something goes (`at=`)
| form | meaning |
|---|---|
| `"nasion"` | at a landmark |
| `rel("nasion", 0, -0.01, 0)` | offset from a landmark |
| `on("head", x=0.031, y=-0.002)` | cast a ray from the front onto `head`'s skin at (x, y); depth comes from the surface |
| `on("head", y=-0.02, z=-0.005, side="L")` | cast from the side (L or R). Also `side="top"` and `side="back"` |
| `…, depth=-0.002` | added to any of the above: move along the surface normal (negative = into the skin) |
| `(x, y, z)` | absolute; allowed, but counted in the report as `absolute placements: n` |

## 5. Commands
| command | what it does | reply (example) |
|---|---|---|
| **session** | | |
| `new(name)` | empty scene | `ok new face` |
| `open(path)` | load a scene: a `.fc` command script, or an old Python scene (becomes one `code` node; needs `--allow-code`). Starts a new history; versions go to `<dir>/runs/<stem>/` | `ok opened face (12 nodes)` |
| `save(path)` · `export(path, res=320)` | `.fc` script (from then on versions are saved next to it) · `.glb` mesh plus a `.json` sidecar with landmarks, sampled curves and the node list (for production steps later) | `ok exported face.glb: 180432 verts, watertight=True` |
| **build** | | |
| `add(name, shape, at=, size=, blend=, op=, mirror=)` | shapes: `sphere ellipsoid box capsule torus`, plus pack parts `eye ear`. `op`: `union` (default), `cut`, `intersect`, `separate` (touching, no blend) | `ok add cheek (ellipsoid, mirror) blend 0.02` |
| `mark(name, at, pin=True)` | landmark; the surface is bent to pass exactly through pinned marks (the engine picks the fit radius) | `ok mark pronasale: base was 3.1mm inside, pinned` |
| `curve(name, through=[…], pin=True)` | named curve through marks and points, for profiles, strokes and later edge loops | `ok curve profile: 16 points, max dev 1.2mm` |
| `stroke(name, brush, along=, amount=, width=, mirror=)` | sculpt along a curve or points, snapped to the skin. Brushes: `crease ridge inflate carve groove` (P2 adds `flatten smooth pinch grab clay layer`) | `ok stroke nasolabial: 38mm long, depth 1.2mm` |
| `code(name, src)` | escape hatch: a custom `def sdf(p)` in Python, e.g. for Voronoi scales. Runs locally, like scene files do today. **Opt-in** (`--allow-code`, §3) | `ok code scales` |
| **edit** | | |
| `set(name, **params)` | change parameters | `ok ear splay -35→-20` |
| `move(name, at=)` · `rename(a, b)` · `delete(name)` | | `ok move eye → on(head, 0.031, -0.002)` |
| **goals: the engine solves** | | |
| `seat(name, contact=0.55)` | search depth, flare and splay until the part sits on its host | `ok seat ear: splay -35→-14°, contact 56%` |
| `solve(param, measure, target)` | e.g. `solve("jaw.size", "width jaw", 0.104)`. The measure names are the ones printed by `measure()` | `ok solve jaw.size 0.050→0.052: width jaw 104mm` |
| `fix(n)` | apply fix *n* suggested in the last reply | |
| **ask: extra detail on demand** | | |
| `measure(a, b)` · `measure(name)` | distance between two names (landmarks, or node centres) · a node's bounding box (the left one if mirrored) and an ear's seating; a mark's position and drift; a curve's length | `nasion->menton 121.0mm` |
| `report()` | the full report (warnings + measurements) | (long) |
| `look(view=, at=, size=)` | image: a view (`front side 3/4 back top all`), zoomed on a named node or landmark. Optional `before=True` shows it side by side with the previous version | `[image] <png path> (side at ear, 0.06 m across)` |
| `tree()` | node list: names, kinds, one line each | `face: cranium, frontal, …` |
| **history** | | |
| `undo(n=1)` · `goto("v004")` · `history()` | undo/goto load an old version as a **new** version (never checked for regressions); undo follows each version's parent, so repeated undo keeps going back. `history()`: one line per version (tag, status, warnings, first change) | `ok undo -> v005` |
| `keep()` | accept the last reverted change anyway (only while the scene is still where it was rejected) | `ok keep v007-rejected (was: +WARN …)` |

## 6. The automatic loop (after every batch)
1. **Rebuild** only the nodes that changed, and anything that depends on them.
2. **Check everything.** That covers all senses from SPEC §6: loose pieces, kinks, seating, scale lint, fit and drift, plus the new `absolute placements` count. Only the region the change touched is re-evaluated; the rest comes from cache.
3. **Auto-frame.** The engine fits the frame to the model's bounding box, so the LLM never sets `FRAME` and clipping can't happen.
4. **Diff** against the last accepted version.
5. **Regression?** That means a new warning, or fit/drift worse by more than 0.5 mm. If so: revert, keep the change as `vNNN-rejected`, and test fixes.
6. **Fix search.** For each new warning, try up to 3 candidate fixes on a copy. Report only the fixes that remove the warning without adding another, with their predicted result:

   | warning | candidate fixes |
   |---|---|
   | loose piece / low seating | `seat(name)`, larger `blend` |
   | kinks on a trim | larger `k`, smoothed outline |
   | mirror seam | larger blend across x=0 |
   | scale lint | the typical value |
7. **Save** the version (`.fc` script + metrics + run log: tokens, time, warnings).
8. **Reply.**

## 7. Reply format
```
ok <one short result per command, joined by " | ">
<changes vs last version: ±WARN …, measurement a->b>  (omitted if nothing changed)
warnings: none | v006 | 0.9s                           (or: N warnings | v006 | 0.9s, then up to 3 WARN lines, new ones first)
```
A regression replaces this with `REVERTED v006: <why>`, the new WARN lines, `back at v005 | 0.9s` and `keep() to accept the change anyway`
(step C adds `fix n:` lines and `fix(1) to apply`). A batch that changes nothing (only queries) makes no version and replies with the results only.
Replies are capped at 600 characters (about 150 tokens): line 1 and the changes end in `... (+N more)`, warnings in `(+N more: report())`.
`report()`, `history()` and `tree()` are listings and are not capped; they come after the version lines.
Queries (`report measure look export`) see the batch's result, even when the batch is then reverted.
Replies are plain ASCII (`->`, `|`, `...`): `→`, `·` and `…` garble on a Windows console (cp950).
**Errors:** `error line 2: set("eer", …): no node "eer" (did you mean "ear"?). Nothing applied.`
Every error names the line, the cause and a fix.

## 8. Transport
- **MCP server** (stdio), the standard way LLM apps call tools. It exposes **one tool**, `formcode(commands: str)`.
  - The tool description *is* the cheat sheet: §4 and §5 compressed to about 300 tokens. One tool costs fewer tokens per session than 20 tools.
  - Replies are text. `look` adds an image (MCP image content, base64 PNG). Errors set `isError: true`.
  - Built on the official Python SDK (`mcp` on PyPI, MIT), as an optional extra `formcode[mcp]`, so the core stays numpy, scipy and pillow.
- **REPL:** `python -m formcode repl [--allow-code]` reads the same commands from stdin. Used for tests and for clients without MCP.
- **Script:** `python -m formcode run face.fc` replays a script. The scene file *is* a command script (final state, one line per node).

## 9. Speed targets and how
| | target | how |
|---|---|---|
| edit → reply | **≤ 2 s** median on the face (today: about 34 s per run) | persistent process; per-node bounding boxes; recompute only the check-grid cells and kink samples inside the changed region (+ blend margin); everything else cached |
| `look` | ≤ 3 s | warm render worker pool; render only the requested view and region |
| fix search | ≤ 3 s extra, only when there is a new warning | candidate fixes evaluated on the changed region only |

## 10. New P1: done when
1. The protocol works through the REPL and through MCP. A real Claude Code session connects and builds through it.
2. **The face is rebuilt through the protocol only:** zero absolute placements for eyes, ears and strokes; every check automatic.
3. Replies are ≤ 150 tokens (except `report` and `look`), and the median edit reply takes ≤ 2 s.
4. A deliberately bad change is reverted automatically, and the suggested fix works.
5. It is measured against session 1's face (7 iterations, tokens used, about 10 defects you found that I missed): tokens, iterations, and defects found by the engine vs. by you.

Already built and reused: warnings and senses (P1 step 1), versions/diff/revert (P1 step 2), the whole kernel (P0).

## 11. Decisions (accepted 2026-10-12)
1. The scene file is a command script (`.fc`).
2. `code()` escape hatch: yes, opt-in via `--allow-code`, off by default (also gates opening `.py` scenes; §3).
3. MCP through the official Python SDK, as the optional extra `formcode[mcp]`.
4. One MCP tool that takes a command string.
