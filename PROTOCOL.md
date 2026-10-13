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
- **File paths stay in a workspace.** The engine has a root (`Engine(root=...)`, `--root DIR`, default: the working directory at start). Every path of `open`, `save` and `export` is resolved against it (`..` and symlinks resolved first) and must stay inside it, else `path outside the workspace <root> (use a path inside it, e.g. "scenes/face.fc")` and nothing is applied. Extensions: `save` `.fc`; `export` `.glb` (+ its `.json` sidecar); `open` `.fc`, or `.py` with `--allow-code`. Versions and `look` images go next to the scene (`<dir>/runs/<stem>/`, or a temp dir before the scene has a path). Same reason: an LLM must not overwrite or read files outside the project.
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
| `seat(name, contact=None)` | search the node's seat knobs until it sits on the rest of the model. Contact = fraction of its surface within 1.5 mm of the model without it. Knobs, least visible first: ear `flare` (0..10 mm), then `splay` (-45..30°), each searched in widening windows around its current value so the chosen look changes as little as possible. Without `contact=` the target is "attached" (ear ≥ 0.45, clear of the 40% warning); with `contact=0.57` it is hit exactly (attached ears measure 55-60%). If the chosen angle can't stay attached, the reply shows how far it had to move; a shape placed with `on()`: its `depth`; any other shape: a move toward the nearest surface, target "just attached" (0.10). `eye` has no seating measure yet (error) | `ok seat ear: splay -35->-14, flare 4.0->3.1mm, contact 56%` |
| `solve(param, measure, target)` | e.g. `solve("jaw.size", "width jaw", 0.104)` or `"jaw.size[0]"`: a bounded 1-D search on one number until the measure hits the target. Measures: `width`/`height`/`depth <node>` (bbox; meters in args, mm in replies), `seating <node>`, `"a->b"` (distance). Not reachable: error with the closest value | `ok solve jaw.size 0.05->0.052: width jaw 104.0mm` |
| `fix(n)` | apply fix *n* of the last REVERTED batch: the rejected batch plus the fix, as one batch through the loop | `ok seat ear: ...` |
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
   A new kink warning counts only if its cluster has ≥ 5 samples and lies in the region the batch changed (the changed nodes' boxes before and after, plus their blend + 5 mm; everywhere if a mark, curve or stroke changed). Kink samples are redrawn on every build, so small or far clusters flicker; they are still shown as warnings.
6. **Fix search.** For each new warning, try candidate fixes **on top of the rejected batch** (so the LLM's intent is kept): at most 4 builds per reverted batch, at most 2 fixes shown. Report only the fixes that remove the warning without adding a regression, with their predicted result:

   | warning | candidate fixes |
   |---|---|
   | loose piece / low seating | `seat(name)`, larger `blend` |
   | kinks on a trim | larger `k`, smoothed outline (not yet: `front_outline` is not an engine node kind) |
   | mirror seam | larger `blend` of a mirrored shape |
   | scale lint | a stroke's `width` / an ear's `size` set to the middle of the typical range |
7. **Save** the version (`.fc` script + metrics + run log: tokens, time, warnings).
8. **Reply.**

## 7. Reply format
```
ok <one short result per command, joined by " | ">
<changes vs last version: ±WARN …, measurement a->b>  (omitted if nothing changed)
warnings: none | v006 | 0.9s                           (or: N warnings | v006 | 0.9s, then up to 3 WARN lines, new ones first)
```
A regression replaces this with `REVERTED v006: <why>`, the new WARN lines, `fix n: <command> -> <predicted result>` lines,
`back at v005 | 0.9s` and `fix(1) to apply | keep() to accept the change anyway` (no fixes found: only `keep() ...`). A batch that changes nothing (only queries) makes no version and replies with the results only.
Replies are capped at 600 characters (about 150 tokens): line 1 and the changes end in `... (+N more)`, warnings in `(+N more: report())`.
`report()`, `history()` and `tree()` are listings and are not capped; they come after the version lines.
Queries (`report measure look export`) see the batch's result, even when the batch is then reverted.
Replies are plain ASCII (`->`, `|`, `...`): `→`, `·` and `…` garble on a Windows console (cp950).
**Errors:** `error line 2: set("eer", …): no node "eer" (did you mean "ear"?). Nothing applied.`
Every error names the line, the cause and a fix.

## 8. Transport
- **MCP server** (stdio), the standard way LLM apps call tools: `python -m formcode mcp [--allow-code] [--root DIR]`
  (`formcode/mcp_server.py`). It exposes **one tool**, `formcode(commands: str)`, backed by one `Engine` per server process;
  `--allow-code` and `--root` mean the same as for the REPL (defaults: code off, root = the working directory).
  - The tool description *is* the cheat sheet: §4 and §5 compressed to 1191 chars, about 300 tokens (chars/4). One tool costs fewer tokens per session than 20 tools.
  - The text reply is `Engine.do()`'s reply unchanged. Each `[image] <path>` line of a `look` also comes back as MCP image
    content (base64 PNG), read only from inside the root or the engine's own temp dir (scenes without a path). A reply
    starting `error line` sets `isError: true`; `REVERTED` does not (it is normal feedback).
  - Calls run one at a time (the SDK runs sync tools in worker threads; a lock serializes them). stdout is the protocol
    channel: the SDK points fd 1 at stderr while serving (spawned workers inherit it), and each call sends `sys.stdout`
    to stderr, so prints in `code()` scenes never reach the wire.
  - Built on the official Python SDK (`mcp` on PyPI, MIT; v2, `MCPServer`), as an optional extra `formcode[mcp]`
    (`pip install -e ".[mcp]"`), so the core stays numpy, scipy and pillow.
  - Connect from Claude Code (stdio; everything after `--` is the server command):
    `claude mcp add --transport stdio formcode -- D:/3dforLLM/.venv/Scripts/python.exe -m formcode mcp --root D:/my/workspace`
    (add `--allow-code` at the end to allow `code()`). A long batch may need a larger `MCP_TOOL_TIMEOUT`.
- **REPL:** `python -m formcode repl [--allow-code] [--root DIR]` reads the same commands from stdin. Used for tests and for clients without MCP.
- **Script:** `python -m formcode run face.fc` replays a script. The scene file *is* a command script (final state, one line per node).

## 9. Speed targets and how
| | target | how |
|---|---|---|
| edit → reply | **≤ 2 s** median on the face (today: about 34 s per run) | persistent process; per-node bounding boxes; recompute only the check-grid cells and kink samples inside the changed region (+ blend margin); everything else cached |
| `look` | ≤ 3 s | warm render worker pool; render only the requested view and region |
| fix search | ≤ 3 s extra, only when there is a new warning | candidate fixes evaluated on the changed region only |

**Measured (step D, 2026-10-12)** with `benchmarks/bench.py face --runs 3` (`benchmarks/face.fc`: face.py as 63 engine
nodes), median of 3 runs, on a busy i9-14900KF (a game was running: absolute numbers are noisy, the ratios hold):

| command | before (HEAD 7b2fbcf) | after |
|---|---|---|
| `open` | 17.0 s | 2.6 s |
| `set` a stroke amount | 17.2 s | 1.7 s |
| `move` a cheek shape | 16.7 s | 2.5 s |
| `add` a 4 mm bump | 17.3 s | 2.1 s |
| `set("ear", flare=0.016)` → REVERTED + fix search | 37.2 s | 6.4 s |
| `keep()` | 17.1 s | 2.6 s |
| `seat("ear")` | 20.9 s | 3.8 s |
| `look(at="ear")` (5 views) / one view | 8.9 s / – | 3.1 s / 1.9 s |
| `look("all")` | 7.8 s | 2.1 s |
| **median edit** | **17.3 s** | **2.5 s** (1.8 s in a quieter run) |

How, now: the measure grid is cached per build stage (`gridcache.py`; base, fit, strokes, eye, ear) and an edit
re-evaluates only the grid points whose stage input changed bit for bit, plus the boxes of changed strokes, so cached
values equal fresh ones exactly (`tests/test_incremental.py`). Part boxes and the ear's seating sample are cached by
the part's signature. The fit sums only the (point, target) pairs inside each target's radius. Evaluations run in a
thread pool (`kernel.peval`), and kink and normal evaluations are batched into a few large calls. Multi-view `look`
uses a warm process pool whose workers keep the last built scene; a single view traces its rays in threads.
Not met yet: the 2 s median under load, `look` with 5 zoomed views, and the fix search (≈3-4 s extra). What dominates
now: the kink sense (~630k sdf evaluations near the surface, 0.7-1.1 s, re-drawn from the whole shell every build, so
it can't be cached without changing its sampling), the build (0.3-0.6 s, mostly the frame bbox search), and a fit
change spreading over the whole surface (a shape edit near a mark changes most fit weights by > 1 µm, so most of the
near-surface grid is redone). The sdf is ~450 numpy passes per call, and threads stop scaling at ~3-5x (GIL and call
overhead). Next steps: fused evaluation (numba, an optional extra), or kink samples on a fixed lattice so they can
be cached (a sense change).

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
