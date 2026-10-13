# HANDOFF — read this first (state as of 2026-10-12, session 2: P0 done)

## What this project is
**formcode**: an open-source, lightweight protocol + engine that lets *frozen* LLMs (Claude, GPT, …, no training)
create detailed, polished 3D models **from nothing** — "spatial understanding in an LLM way".

The user's non-negotiables:
- The LLM is the creator. **No 3D-gen models** (Meshy, Tripo, TRELLIS), **no shape templates/libraries** doing the work.
- **Open source, lightweight**: pure Python + numpy core, GPU optional.
- **Borrow existing languages/idioms** rather than invent syntax.
- Goal is **general** (characters, creatures, props, vehicles, environments) at polished/production quality,
  including **sculpting**; a detailed character is the hardest showcase.
- Token-efficient and fast.

Image models (ComfyUI) are acceptable only as an *input* (e.g. a reference sheet for testing), not as the creator.

## Where we are
- **Session 1** was research and experiments: findings in `experiments/SUMMARY.md`, per-iteration log in `experiments/sdf_lab/FINDINGS.md`.
- **Session 2** settled the 4 decisions and finished **P0**:
  - Python vocabulary, not a DSL.
  - `SPEC.md` and `CLAUDE.md` rewritten. `SPEC.md` is now the authority; `PLAN.md` is the rationale.
  - Apache-2.0 (`LICENSE`), name `formcode`.
  - Blender M0 scaffolding moved to `later/`.
- The **`formcode` package** (`kernel.py`, `render.py`, `senses.py`, CLI) is the lab code ported verbatim, plus:
  - a scene-level `FRAME` sets the render, export and report bounds;
  - one merged report, with every sense on every run;
  - registries reset per scene load;
  - chunked mesh export (fixes an out-of-memory error).
- `examples/` holds the final scenes of all 4 experiments.
  - All 20 views rendered **pixel-identical** to the lab code.
  - Face and anime now use a wider frame `(0,-0.03,0,0.18)`, so the neck is no longer clipped and both meshes are **watertight**.
  - Golden tests cover each example's report text and 64 px renders. 13 tests pass, and a 2 mm ear change makes them fail.
- `refcompare.py` (anime reference compare) is **not ported**. Its constants are tied to one reference sheet, and P1 should replace it with auto-extracted outlines (lesson 8).

### The experiments (all pure numpy SDF, written by the LLM)
| exp | what | result |
|---|---|---|
| 01 | dragon head, primitives + blends | recognizable, mid-detail; Voronoi code = thousands of scales |
| 02 | human body, skeleton-first | mannequin body; face = caricature with ellipsoids |
| 03 | realistic face, **anatomical vocabulary** (landmarks + profile curves + sculpt verbs) | plausible statue-like head; best result |
| 04 | anime head from an Anima reference sheet + reference-compare sense | outlines matched ~1–2 mm, but looks like "E.T." |

### Key lessons (to build into the engine)
1. **Vocabulary is the bottleneck, not LLM knowledge.** Use names, landmarks, curves and strokes, never raw coordinates.
2. **Text senses beat LLM vision** for relations and measurements. The best sense was "base is X mm off at landmark Y".
3. **LLM vision misses** attachment, holes vs bumps, anything off-view, and its own mistakes. The user caught ~10 defects the model missed.
4. **Relative placement** (snap, attach, swing around root, depth from skin) kills whole bug classes.
5. **The LLM knows *where*, not *how big*.** A scale lint helps.
6. **Every sense must run on every iteration.** Two-view silhouette matching hid floating ears and ridges (Goodhart).
7. **Senses must know what they measure** (hair vs skin, neck behind chin, contact % is wrong for protruding parts).
8. **The engine measures references better than the LLM reads them** (≈7 mm reading error), so auto-extract outlines.
9. **Most tokens went to tooling bugs, hand-typed coordinates, image re-reads and whole-file rewrites**, not modelling.
10. **Stylized looks (anime) live largely in texture**: painted eyes and brows, and hair. Geometry alone looks alien.
11. **LLM sculpting is barely explored.** Prior art: ViSculpt 2026 (GUI mouse actions). Our "stroke = one line of text
    on anatomical curves" approach appears new.

## P1 progress (SPEC §9)
**Reframed on 2026-10-12 (user):** P1 is now the **always-on engine + protocol** in `PROTOCOL.md` (decisions accepted).
The engine does the heavy lifting: after every change it rebuilds, checks, versions, reverts, and tests fixes.
The LLM only says what, where and what it's called. Steps 1 and 2 below are reused inside the engine loop.

**Step 1, done: warnings at the top of the report** (SPEC §6). Results:
- loose pieces are rechecked 8x finer, so the dragon's 9 and the human's 2 thin-neck fragments are no longer false alarms;
- a new kink sense with an intended-edge registry and cause hints;
- clipping, seating and scale lint are now warnings.

The face reports `warnings: none`. The sense found things nobody had noticed:
- anime: front_outline trim creases (the side "stripes"), plus a stuck-on ear;
- human: a mirror seam along the crotch and spine;
- dragon: the neck is clipped at the back of FRAME.

Following the anime hint on a scratch copy (spline-resampling `FRONT_OUTLINE`) cut kink samples 62 → 14 and removed most of the banding.
The example itself is unchanged: it is the frozen exp04 result.

**Steps 2 + 3, done: versions, diff report, auto-revert, run log** (SPEC §7, `formcode/versions.py`).
- `senses.measure()` returns the report as data; `report()` formats it (goldens unchanged).
- Every render run saves `runs/<scene>/vNNN.py` + `.json`; the JSON is the run log (tokens, warnings, seconds, all metrics), so step 3 is merged into it.
- A new warning key, or worst profile dev / landmark drift sum worse by >0.5 mm, reverts the scene and keeps the run as `vNNN-rejected`; `--keep` accepts.
- Checked on a scratch face copy: ear `flare` 0.004 → 0.010 was accepted (`seating ear 58%->45%`), → 0.016 was reverted (`+WARN seating ear`, 33%).
- Open: a seating drop that stays above the 40% warning is not a regression; kink keys include `near <landmark>`, so a cluster that moves to another landmark counts as new.

**Engine step A, done** (`formcode/engine.py`, `tests/test_engine.py`): `Engine().do(batch) -> reply`, `Scene` (named nodes,
`fc()`, `absolute_placements()`, `build()`), `parse()` (ast literals + `on()`/`rel()`, never executed), `python -m formcode repl`;
`render.load` builds `.fc` scripts. Choices to revisit:
- default `blend` is 0.01 m; `code(name, src)` also takes `op=`/`blend=`; a legacy `.py` becomes one `code` node holding the whole file.
- marks resolve in stage 2 against the base; a shape in stage 1 may use a mark only if that mark doesn't sit on the body.
- fit radius per target = 4x nearest-neighbour target distance, clipped to 5-20% of the base size; curves are sampled at 1/3 of the mean control-point gap (step F tunes both).
- auto-frame: a shrinking-grid bbox search from an 8 m cube, FRAME half = 0.55 x the largest extent (face: 0.173 vs the hand-set 0.18; still `warnings: none`).
- `rename` updates references in `at/a/b/through/along`; `delete` of a referenced node fails the batch at build.
- every changed batch rebuilds everything and runs the full `measure()` (face ~10 s): step D's job.

**Engine step B done** (`formcode/session.py`: `Engine`, the loop, history + query commands, REPL; `versions.History`;
`tests/test_engine_loop.py`). After every changed batch: build, `measure()`, `versions.verdict` (the step-2 diff and
regression rule, shared with the file CLI) vs the last accepted version, then save `vNNN` or revert and keep `vNNN-rejected`.
Versions live in memory and, once the scene has a path (`open`/`save`), in `<dir>/runs/<stem>/vNNN.fc` + `.json`.
`undo/goto/keep/history`, `report/measure/look/export` work; replies follow PROTOCOL §7 in ASCII (cp950), capped at 600 chars.
Code execution is opt-in: `Engine(allow_code=True)` / `repl --allow-code`, else `code()`, `.py` open and `.fc` files with code fail.
Found while timing the face (12 s per batch):
- **false regressions from the kink sense:** `set("bump", size=0.006)` on a 4 mm bump at the cheek was reverted for
  `+WARN kinks midline near gnathion` (a 3-sample cluster far from the edit). The 3000 kink samples are re-drawn from the
  whole shell, so any edit reshuffles them and small clusters come and go. Step C/D should make the kink regression
  local (only clusters near the changed region) or need a minimum cluster size.
- **the auto-frame face export is not watertight** (res 320, half 0.173: 2 non-manifold edges at x=±29, y=-50, z=-55 mm,
  behind the jaw), while the CLI with the hand-set FRAME 0.18 is watertight. A pinch the coarser-aligned grid hits.
- undo/goto make a new version (status accepted, never checked) and walk `parent` links, so repeated undo keeps going back.

**Engine step C done** (`formcode/goals.py`, `tests/test_engine_goals.py`): `seat(name, contact=None)`, `solve(param, measure, target)`,
fix search + `fix(n)`, and a workspace root for file paths.
- contact = `senses.seating` generalized (same fixed grid, evaluated only in a box around the part, so the ear's number
  equals the sense's) of a node vs the model without it (`Scene.build(skip=name)`). Ear: coordinate passes over splay
  then flare (`minimize_scalar` bounded, 2 rounds), trials call `kernel.ear` on the same host. Shapes: `on()` depth or a
  move toward the nearest surface, `brentq` to contact 0.10. Replies report the contact of the loop's own build.
- `solve`: `brentq` when the bounds bracket the target, else bounded `minimize_scalar`; each trial rebuilds a copy.
  Default bounds x0/4..4x0 (or the seat-knob bounds). `_extent` (measure/look/solve) now refines each side to the surface (~um).
- fix search: catalogue in `goals._candidates`, tried on the rejected batch, at most 4 builds, 2 shown; `fix(n)` replays the
  checked scene. `front_outline` trims are not an engine kind, so no trim fix yet.
- kink regressions: a new cluster counts only with >= 5 samples and inside the changed nodes' boxes (+ blend + 5 mm).
  Fixed the step-B false revert (reproduced on a small scene: a 3-sample cluster at a far bud flickers with every nose edit).
- `Engine(root=)` / `repl --root DIR`: open/save/export paths resolved and kept inside the workspace, extensions checked.
- Timings: small head scene seat ear ~1 s on top of a 1.4 s batch, fix search ~3.6 s (2 candidates). Face copy (its ear
  as an engine `ear` node): batch ~9 s, `seat("ear")` +2 s, fix search +10.5 s (one candidate = seat + full build + measure).
- Open: seat's bounded search may land far from the start (face: splay -30 -> 20.3, flare kept at 10 mm); a "closest to
  the current value" preference would read better. Fix candidates are full rebuilds (step D: region-only checks).

**Engine step D done (speed)**. New: `formcode/gridcache.py`, `benchmarks/face.fc` (face.py as 63 native nodes: 12
blockout shapes, 31 marks, profile + back curves, 16 strokes, eye, ear) and `benchmarks/bench.py`, plus `tests/test_incremental.py`.
Medians of 3 runs on a busy machine (full table in PROTOCOL §9):

| | before | after |
|---|---|---|
| face: median edit (stroke / cheek move / bump / keep / seat) | 17.3 s | 2.5 s (1.8 s quieter) |
| face: flare 0.016 → REVERTED + fix search | 37.2 s | 6.4 s |
| face: `look("all")` / `look(at="ear")` (5 views) / one view | 7.8 / 8.9 s / – | 2.1 / 3.1 / 1.9 s |
| face: `open` | 17.0 s | 2.6 s |
| small scene (`bench.py small`): median edit | 4.3 s | 0.74 s |
| face export (res 320) | ~43 s | ~15 s |
| `pytest -q` | 87 s (57 tests) | 56-66 s (58 tests) |

- Profile before: `kernel.fit` evaluation was 58% of all time (a dense 48-target block per point), then the kink sense
  (24%) and the per-view process pools, each rebuilding the scene. After: the kink sense (0.7-1.1 s) and the build (0.3-0.6 s) dominate.
- What changed: the fit sums only the (point, target) pairs within each target's radius (KD tree, pairs sorted, so it's
  exact in any chunking). A thread pool runs the evaluations (`kernel.peval`), and kink/normal evaluations are batched. Kernel stages expose
  `stage = (key, sig, apply(v, p), elems)`, `Scene.build` lists them in `mod.STAGES`, and the engine's `GridCache` redoes a
  stage only where its input changed bit for bit, or in the boxes of changed strokes. Part boxes and the ear's seating sample are cached by
  signature. The base bbox for the fit radii is memoized. All strokes snap onto the surface in one call. Multi-view `look` runs in a warm process pool
  (workers keep the last built scene); a single view traces its rays in threads.
- Exactness: cached grid values equal fresh ones bit for bit, so the measure is identical. Tested on a seeded random 12-step
  edit sequence. Goldens are untouched and pass (the fit's summation order changed only in the last bits).
- Fixed a step-C bug: `seat` on an ear with a knob past its range (flare 16 mm > 10 mm) made an empty search window,
  and the ValueError escaped the fix search.
- Tried and dropped: snapping the auto-frame to a lattice, so the grid survives frame-moving edits (e.g. an ear splay
  that shifts the frame by 0.4 mm). It changed the legacy face's grid, and the re-drawn kink samples then showed a
  3-sample flicker cluster (test_open_legacy_face). Such edits now redo the whole grid.
- `benchmarks/face.fc` vs `examples/face.py`: the same drift and profile numbers within 0.1 mm, and seating ear 58%. It
  looks the same apart from shading (a .fc can't set EPS/STEP/AO), a slightly fuller top-back of the cranium
  (heuristic fit radii vs face.py's 0.045 m on the back curve), and one extra warning: `scale: mouth_slit length 58.4mm`
  (the strokes snap onto a slightly different fitted skin). Its engine export at res 320 is watertight.
- Open: a 2 s median under load. The kink sense can't be cached while its samples are re-drawn from the whole shell (a
  fixed-lattice sampling would make it cacheable, but it changes the sense). A fit change spreads over the whole surface. Threads cap at
  ~3-5x (GIL); numba would be the next step (an optional extra, the user decides). Export still rebuilds the scene per slab worker.

**Build plan for the engine (PROTOCOL.md §10 is the finish line):**
- A. Core: a scene as named nodes, the command parser (`ast`, never exec), placement (`on`/`rel`/landmarks), build/edit commands, `.fc` save/open, REPL.
- B. Automatic loop: checks, versions and revert per batch; auto-frame; the reply format; errors with fix hints.
- C. Goals and fixes: `seat`, `solve`, fix search.
- D. Speed: persistent caches, dirty-region checks, warm render pool; `look`, `measure`, `report`.
- E. MCP server (`formcode[mcp]`, one tool).
- F. Rebuild the face through the protocol and measure against session 1.

**Rough edges:**
- bounding boxes print in cm (2 decimals in meters);
- the landmark labels clutter the side view;
- kinks below 4 degrees are missed (gentle facets still shade visibly);
- the kink sense adds about 4 s to a face report (34 s total with render).

## How to run things (Windows 11, Git Bash)
- venv: `.venv` (Python 3.12, numpy, pillow, scipy, scikit-image, trimesh, pytest), created with `uv venv --python 3.12`;
  `uv pip install --python .venv/Scripts/python.exe -e .` installs the package.
- Package (installed editable): `.venv/Scripts/python.exe -m formcode examples/face.py out.png [cx,cy,cz,half]`.
  An output path ending in `.glb` exports a mesh over the scene's FRAME. Tests: `.venv/Scripts/python.exe -m pytest -q`.
- Old lab (frozen): `cd experiments/sdf_lab && ../../.venv/Scripts/python.exe lab.py dragon.py out.png` / `facelab.py face.py out.png`.
- Anime compare: `cd experiments/anime_ref && ../../.venv/Scripts/python.exe refcompare.py anime.py out.png`.
- Viewer: `cd experiments/sdf_lab/runs && ../../../.venv/Scripts/python.exe -m http.server 18765 --bind 127.0.0.1`,
  then open `http://127.0.0.1:18765/viewer.html`. The user has **no default app for .glb**; port 8765 is blocked (Hyper-V range).
- ComfyUI (only if the user asks for image generation): start from its own folder
  `C:/Users/erede/Downloads/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable` with
  `./python_embeded/python.exe -s ComfyUI/main.py --windows-standalone-build --disable-auto-launch` (port 8188);
  stop it afterwards (kill the process tree). Anima: UNet `anima_aestheticV11.safetensors`, CLIP `qwen_3_06b_base` type `stable_diffusion`,
  VAE `qwen_image_vae`, 30 steps, cfg 4, euler/simple (from ComfyUI's `image_anima_base_v1` template). Script: `experiments/anime_ref/gen.py`.
- Blender 5.1.2 is installed at `C:/Program Files/Blender Foundation/Blender 5.1/blender.exe` (also 4.3).

## Gotchas hit this session
- The system code page is **cp950**: always `open(..., encoding="utf-8", newline="\n")`, and set `PYTHONIOENCODING=utf-8` when printing unicode.
- `ProcessPoolExecutor` on Windows can't spawn from stdin scripts (`python - <<EOF`). Use files or `-c`.
- When `facelab.py` runs as `__main__`, the scene imports a second `facelab` module, so read registries from `import facelab`, not globals.
- SDF traps: steep displacements break sphere tracing (use true cuts); small-radius RBF fits make bumps; sparse splines cut corners.
