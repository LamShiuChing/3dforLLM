# Protocol / engine plan — rationale (2026-10-12)

> **Decided in session 2:** Python vocabulary, Apache-2.0, name `formcode`, Blender files moved to `later/`.
> The spec is now `SPEC.md`; milestones and status live there. This file keeps the reasoning.

Based on `experiments/SUMMARY.md`. This supersedes the Blender-first plan in `SPEC.md`. That file will be rewritten once we agree.

## Thesis
An LLM can create detailed 3D from nothing **if the vocabulary matches how it thinks**: names, relations, curves and strokes, never raw coordinates. The engine also has to **see for it** in text.

Constraints:
- Open source and lightweight: pure Python plus numpy, with GPU optional.
- No generative models and no shape templates.

## What the engine must provide

### 1. Language: borrow, don't invent
- A **small Python vocabulary**. The LLM wrote it fluently in every experiment, with no new syntax to learn.
- A terse DSL only if token measurements later show Python is too expensive. *(Decision needed.)*
- **Everything is named**: parts, landmarks, curves, strokes. Edits and feedback refer to names.

### 2. Geometry kernel (operations, not content)
- SDF primitives, smooth booleans, mirror, patterns (Voronoi and others, aligned to the surface flow).
- **Skeletons**: named joints, with flesh hung on bones.
- **Landmarks and profile curves → fit**: the surface passes through them. Wide radius for sparse points, curvature-aware splines.
- **Relative placement by default**:
  - `attach` and `snap` (ray cast onto a host surface)
  - `on(part, side, t)`
  - depth measured from the surface

  Absolute xyz is the exception.
- **Sculpt brushes as text strokes** along curves between landmarks or on named regions. The engine guarantees bounded slopes.

  | Brush | Status |
  |---|---|
  | crease, ridge, inflate, groove, carve | have |
  | flatten, smooth, pinch, grab, clay, layer | need |

### 3. Senses: text first, always on, warnings at the top
- **Warnings**:
  - floating parts
  - **seating** (contact %)
  - interpenetration between parts not meant to blend
  - clipping
  - symmetry
  - scale lint
- **Fit report**: base vs landmark offset in mm, ranked. This was the best sense.
- **Measurements in domain units**: head units for bodies, mm for faces.
- **Per-stroke report**: length, max depth, which landmarks moved.
- **Queries**: the LLM asks ("is the handle attached?", "jaw width?") and the engine answers.

### 4. Vision: on demand, full coverage
- 6 fixed views including the back, camera-relative light, sparse labels.
- **Zoom by region name**.
- **Pairwise before/after**.
- Images go to the LLM only for overall impression.

### 5. Loop control
- Versions and checkpoints, a **diff report** per change, and **auto-revert** when warnings or fit error get worse.

### 6. Vocabulary packs (names and typical sizes, not shapes)
- Anatomy: face and body landmark names, typical sizes for the scale lint.
- Later: vehicles, furniture, architecture.
- The LLM still places everything itself. A pack only gives it names and a sanity check.

### 7. Output
- Mesh export with a watertight check.
- Later: clean topology, UVs and rig, optionally through a Blender worker behind a process boundary.
- MCP server so any LLM client can use it.

### 8. Evaluation harness (proof that it works)
- A fixed task set: face, body, creature, prop, vehicle, garment.
- Metrics:
  - defects found by a human vs. by the engine
  - iterations
  - tokens
  - regressions
- Baselines: plain SDF/OpenSCAD with no senses.

## Proposed milestones (each ends with an experiment and a measurement)

| # | Milestone | Done when |
|---|---|---|
| P0 | Turn the lab into a `formcode` core package (kernel, renderer, senses) with tests | the three experiments reproduce from the package |
| P1 | Relative placement, always-on warnings, compare/revert | the face rebuilt with zero absolute depths; auto-revert triggers on a seeded regression |
| P2 | Full sculpt brush set with per-stroke reports | a nose/ear redo with no jagged junctions |
| P3 | Anatomy pack: body landmarks, muscle layer (muscles between joints, skin over them), scale lint | full human with face and visible musculature, judged against exp02 |
| P4 | Fabric: garment panels and seams → small cloth sim on the body | a shirt with physically plausible folds |
| P5 | Generality: one hard-surface asset (chair or car) and one creature | same protocol, no new core concepts needed |
| P6 | Production + MCP server + eval harness | an external LLM session builds an asset end to end; tokens are logged |

## Open decisions
1. Language: Python vocabulary (recommended) or a custom DSL?
2. Rewrite `SPEC.md` and `CLAUDE.md` around this plan?
3. License (MIT / Apache-2.0) and project name.
4. What to do with the M0 Blender worker files: keep them for P6 or delete.
