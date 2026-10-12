# formcode — Spec (v1, 2026-10-12)

Replaces the Blender-first v0 spec (still in git history at `b81d074`). Rationale and evidence: `experiments/SUMMARY.md`; roadmap discussion: `PLAN.md`.

## 1. Goal
An open-source, lightweight protocol and engine that lets **frozen** LLMs (no training) create detailed, polished 3D models **from nothing**.
- The LLM is the creator. **No 3D-generative models and no shape templates or libraries** do the modeling. Image models may only supply *inputs*, such as a reference sheet.
- General: characters, creatures, props, vehicles and environments, including sculpting. A detailed character is the hardest showcase.
- Lightweight: pure Python + numpy core (scipy and pillow too). GPU optional. License Apache-2.0.
- Token-efficient and fast.

## 2. Principles (each one came from an experiment)
1. **Borrow a language, don't invent one.** The interface is a small **Python vocabulary**. A terse DSL comes only if token measurements show Python is too expensive.
2. **The vocabulary is the bottleneck, not LLM knowledge.** Use names, landmarks, curves and strokes. Raw coordinates are the exception, and raw vertex lists are never exposed.
3. **Relative placement by default** (snap, attach, depth from skin). The LLM knows *where* better than *how big*, so scale gets linted.
4. **Text senses first, and every sense runs every time.** Images are only for an overall impression and come on demand with full coverage (including the back).
5. **Determinism.** All randomness is seeded. Examples have golden tests.

## 3. Architecture
```
scene.py (LLM-written Python)  ──>  formcode.kernel   vocabulary: SDF ops, curves, fit, sculpt, anatomy parts
          `from formcode import *`  formcode.render   load scene, sphere-traced view grid, marching-cubes export
          defines sdf(p)            formcode.senses   one text report: geometry + attachment + lint + fit
                                    python -m formcode  CLI
later/  Blender worker for production steps (topology/UV/rig), P6. Behind a process boundary; only it imports bpy.
```
A model is a signed distance function `sdf(p)`: `p` is an (N,3) array in meters, it returns (N,) distances, and negative means inside.
Axes: y up, +z = the subject's front (toward the default camera), x = the subject's left. Names ending in `_L` mirror to `_R`.

## 4. Scene contract
A scene is a Python file. It must define `sdf`. It may define:

| name | meaning |
|---|---|
| `FRAME = (cx, cy, cz, half)` | cube that is rendered, exported and measured. Default `(0, 0, 0, 1)`. Must contain the whole model, or the report says `clipped_at_bound=True` and the mesh is open. |
| `PARTS = {name: sdf}` | per-part bounding boxes; floating pieces are attributed to a part |
| `LANDMARKS = {name: (x,y,z)}` | drawn on renders; drift from the surface is reported |
| `PROFILE = [landmark names]` | midline curve; max deviation per segment is reported |
| `FIT_CORRECTIONS = {name: m}` | how far the blockout was from each landmark before `fit()` |
| `VIEW_NAMES` | subset of `front, side (from +x), 3/4, top, back, side (from -x), bottom` |
| `EPS, STEP, MAX_STEPS, AO` | sphere-tracer knobs for small or detailed models |

A scene may edit `TYPICAL` (the scale-lint table). The edit is reset before the next scene loads.

## 5. Vocabulary (`formcode.kernel`)
- **Primitives:** `sphere(p,c,r)`, `ellipsoid(p,c,r)`, `box(p,c,half,round)`, `capsule(p,a,b,ra,rb)` (round cone), `torus(p,c,R,r)`, `polygon2d(q,V)`.
- **Combining:** `smin/smax(a,b,k)` (smooth union/intersection), `cut(a,b,k)`, `mirror_x(p)`, `rot(p,axis,deg,center)`.
- **Detail:** `noise3(p,freq,seed)`, `voronoi(p,cell,seed)` → (F1,F2), `near_surface(d,p,fn,band)`.
- **Curves and surface queries:** `spline(pts,step)` (Catmull-Rom), `dist_to_curve`, `on_surface(f,pts)` (snap), `cast(f,origin,dir)` (ray hit), `with_mirrors(L)`.
- **Fitting:** `fit(base, targets, radius)` → `(sdf, corrections)`: the surface is bent to pass exactly through every landmark or curve sample.
- **Sculpt strokes:** `sculpt(f, ops)`. Each op is `(name, kind, points, amount, width, sym)`, with kind in `crease | ridge | inflate | carve | groove`. Curves are snapped to the surface and the op is linted against `TYPICAL[name]`.
- **Silhouette:** `front_outline(f, [(y, halfwidth), ...], k)` trims to a front-view outline.
- **Anatomy parts** (placed relative to the skin; P3 moves them to a vocabulary pack):
  - `eye(f, center=(x,y), r, en, ex, up, lo)`: the depth comes from the skin.
  - `ear(f, tragion=(y,z), flare, size, splay)`: x is snapped onto the head. The ear registers for the seating sense.

## 6. Senses (`formcode.senses.report`)
`senses.measure(mod)` returns every sense as data (JSON-ready dict: `warnings` as (key, text) pairs with key = kind + subject, `bbox`, `components`, `symmetry_err`, `clipped`, `seating`, `kinks`, `fit`/`drift`/`profile` in mm, and the measurement `lines`); `report(mod)` = `text(measure(mod))`.
One text block, every check on every run. **Warnings come first**, then `--`, then measurements:
```
WARN loose piece (<part>) at x,y,z vox=n: not connected to the body
WARN clipped at FRAME +x -z: the model extends past it (open mesh); grow or move FRAME
WARN seating <part>: only n% in contact: stuck on / floating
WARN scale: <name> <measure> n mm (typical a-b)
WARN kinks: n creased samples, <midline|both sides|left|right> [near <landmark>|on <part>], box x,y,z..x,y,z mm  (cause hint)
warnings: none                             (when there are none)
--
ALL: min=… max=… size=… vol=…            (+ one line per PART)
components=n (k attached by necks thinner than the grid)
symmetry_err(x)=…  clipped_at_bound=…
seating <part>: n% of surface in contact
kinks: n of m surface samples creased (k at intended edges)
fit: base was off by <landmark> ±n mm, …  (top 5)
landmark drift (>0.5mm, +out/-in): …
profile max dev: <a>-><b> ±n mm, …        (worst 4)
```
- **Loose pieces:** every component that is not the main body gets rechecked on a grid 8× finer around the gap. Pieces joined by a thin neck are counted as attached and don't produce a warning.
- **Kinks** (C1 creases): on 3000 surface samples, the normal is traced along short lines across the surface. A kink is where one tiny step carries most of the turn; smooth curvature spreads it evenly. Kinks are clustered with mirrored pairs merged, and up to 3 clusters are warned. Cause hints:
  - *mirror seam*: the cluster sits on x=0;
  - *front_outline trim*: the cluster sits on a trim polygon.
- **Intended edges:** kernel functions that make sharp edges on purpose register themselves in `kernel.EDGES` (groove and carve strokes, `eye`, `ear`), and kinks there are not warned. When `near_surface` detail is in use, kinks are expected everywhere, so they are counted but not warned.

Renders: a labeled grid of `VIEW_NAMES` with a camera-relative light. `render.compare(a, b, out)` puts two images side by side.

## 7. CLI
```
python -m formcode scene.py out.png [cx,cy,cz,half] [--keep]   render grid + report + version (formcode.versions.run)
python -m formcode scene.py out.glb [res]                      marching-cubes mesh over FRAME (needs formcode[mesh]); no version
```
**Versions.** Every render run saves `<scene_dir>/runs/<stem>/vNNN.py` (source) and `vNNN.json` (the run log: version, status baseline|accepted|kept|rejected, time, chars, tokens = chars//4, seconds, warning count, `metrics` = `measure()`). An unchanged scene makes no new version. After the report:
```
vs v003: +WARN kinks midline near nasion | -WARN seating ear | worst profile dev 2.0->1.4mm | landmark drift sum 7.8->6.1mm | seating ear 32%->58%
vs v003: no measurable change
run v004: ~T tokens, W warnings, S s
```
Values are listed only when their printed form changes (0.1 mm, 1%). Landmark drift sum = Σ|drift| over non-`_R` landmarks.
**Auto-revert.** A run is a regression vs the previous accepted version if it adds a warning key (kind + subject, e.g. `loose piece (moon)`, `seating ear`, `kinks midline near nasion`; counts in the text don't matter), or the worst |profile dev| or the landmark drift sum grows by more than 0.5 mm. Then it is saved as `vNNN-rejected`, the scene file is restored to the previous accepted source, and the output starts with
```
REGRESSED vs v003 -> scene reverted (kept as v004-rejected). Re-run with --keep to accept.
why: +WARN loose piece (moon)
```
The render shows the rejected version. `--keep` accepts a regression (status `kept`); run right after a revert on the untouched scene, it restores the rejected source and accepts it.

## 8. Repo layout
```
formcode/     kernel.py render.py senses.py versions.py __main__.py
examples/     dragon.py human.py face.py anime.py   (final scenes from experiments 01–04; runs/ is git-ignored)
tests/        test_kernel.py, test_versions.py, test_examples.py + golden/ (report text + 64 px renders per example)
experiments/  session-1 research: lab code, every iteration, findings (frozen; not imported by the package)
later/        Blender M0 scaffolding, parked for P6
```

## 9. Milestones (each ends with an experiment and a measurement)
| # | Milestone | Done when |
|---|---|---|
| **P0 ✓** | Lab → `formcode` package (kernel, renderer, senses) with tests | the experiments reproduce from the package (all 20 views pixel-identical to the lab; golden tests) |
| P1 | Relative placement everywhere, always-on warnings at the top, versions + diff report + auto-revert | the face rebuilt with zero absolute depths; auto-revert triggers on a seeded regression |
| P2 | Full sculpt brush set (flatten, smooth, pinch, grab, clay, layer) with per-stroke reports and bounded slopes | a nose/ear redo with no jagged junctions |
| P3 | Anatomy pack: body landmarks, muscles between joints, skin over them, scale lint in head units | full human with face and visible musculature, judged against exp02 |
| P4 | Fabric: garment panels + seams → small cloth sim on the body | a shirt with physically plausible folds |
| P5 | Generality: one hard-surface asset and one creature | same protocol, no new core concepts |
| P6 | Production (topology/UV/rig via `later/` Blender worker) + MCP server + eval harness | an external LLM session builds an asset end to end; tokens logged |

Eval metrics (from P1 on): defects found by a human vs. by the engine, iterations, tokens, regressions. Baseline: plain SDF with no senses.

## 10. Conventions
- Every new vocabulary word ships with: a test, use in an example, its line in the report (if it has one), and an error message with a fix hint.
- Every new sense runs by default. A sense that must be switched on will be forgotten.
- Changing the vocabulary, scene contract, report format or CLI → update this file in the same change.
- Windows: the system code page is cp950. Open files with `encoding="utf-8"`. Scripts run by `ProcessPoolExecutor` must be files, not stdin.
