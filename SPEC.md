# formcode — Project Spec (v0)

> Working name. A 3D modeling engine built **only for LLMs**: no UI, a terse text language, cheap text feedback, Blender as a headless backend.

---

## 1. Goal

Let an LLM build good 3D assets (characters, props, environments, materials) by working in representations it handles well: **names, proportions, graphs, 2D sections, and short code**. The engine does all of the actual 3D work.

**First showcase target:** a clean, quad-remeshed, rigged anime-style humanoid (adult proportions), exported to GLB/VRM, built entirely by an LLM through the tool.

### Non-goals (for now)
- Any human-facing UI or viewport
- Forking or modifying Blender source
- Real-time performance or a game engine
- Raw vertex editing by the LLM

---

## 2. Design principles

1. **The LLM never touches vertices.** It writes intent (primitives, skeletons, sections, proportions, named regions). The engine produces geometry.
2. **The dimension ladder.** The LLM works at 0D (names and numbers), 1D (skeletons and curves), and 2D (cross-sections and silhouettes). Only the engine works in 3D.
3. **Token-efficient I/O.** Terse DSL in, compact diffs out. A typical tool response should be ≤150 tokens, and a full scene summary should be ≤500 tokens.
4. **Text feedback first, images last.** Measurements, health checks, and silhouette-vs-reference scores come back as text. Small renders are optional and on demand.
5. **Deterministic and undoable.** Seeded randomness, named checkpoints, and `undo N`. The same script always produces the same result.
6. **Our engine owns the state; Blender is a replaceable backend.** Blender runs in a separate worker process.
7. **Grow the language from observed LLM failures**, not from guesses.

---

## 3. Architecture

```
LLM (Claude Code, etc.)
  │  DSL text
  ▼
MCP server  ─ tools: run, inspect, measure, check, view, compare, undo, checkpoint, export
  │
  ▼
formcode engine (Python, pure — no bpy import)
  ├─ dsl/       parser → AST → op list (with line numbers for errors)
  ├─ scene/     scene state, op history, checkpoints, diffs
  ├─ sdf/       SDF primitives, smooth booleans, evaluation grid, meshing
  ├─ skel/      skeleton, proportions (head units), sections, lofting   [M6]
  ├─ mesh/      trimesh/manifold ops, health checks, measurements
  ├─ feedback/  measurements, silhouettes, reference comparison
  └─ backend/   client for the Blender worker (JSON-RPC over stdio)
  │
  ▼
Blender worker (separate process: `blender -b --python worker.py` or the `bpy` wheel)
  └─ quad remesh (QuadriFlow), modifiers, rigging, materials, render, GLB/VRM export
```

**Why a separate worker process:** it isolates crashes and Blender's Python-version constraints, keeps Blender's GPL code at a process boundary, and lets the core engine run and be tested without Blender installed.

---

## 4. Repo layout

```
formcode/
├─ CLAUDE.md                 # short instructions for coding sessions
├─ SPEC.md                   # this file
├─ pyproject.toml
├─ formcode/
│  ├─ dsl/        parser.py, ast.py, errors.py
│  ├─ scene/      state.py, history.py, diff.py
│  ├─ sdf/        primitives.py, ops.py, grid.py, mesher.py
│  ├─ skel/       skeleton.py, proportions.py, sections.py, loft.py
│  ├─ mesh/       ops.py, health.py, measure.py
│  ├─ feedback/   silhouette.py, compare.py, report.py
│  ├─ backend/    client.py, protocol.py
│  └─ server/     mcp_server.py
├─ blender_worker/
│  └─ worker.py               # runs inside Blender; only file that imports bpy
├─ examples/                  # .form scripts (snowman.form, humanoid.form, ...)
├─ tests/
│  ├─ test_dsl.py, test_scene.py, test_sdf.py, test_feedback.py
│  └─ golden/                 # expected measurements / hashes per example
└─ out/                       # generated GLB/PNG (gitignored)
```

---

## 5. The DSL (v0)

### Syntax
- One op per line. `;` also separates ops. `#` starts a comment.
- Form: `<cmd> [name] [positional...] key=value ...`
- Vectors use commas with no spaces: `at=0,1.2,0`
- Units are meters by default. Inside a skeleton or body context, the suffix `h` means **head units** (`1.35h`).
- Names are `snake_case`. Side suffixes `_L` and `_R` are recognized by `mirror`.
- Every op returns a compact result line, and errors cite the line number plus a fix hint.

### v0 commands (milestones M1–M2)
| Command | Example | Meaning |
|---|---|---|
| `sphere` | `sphere head r=.12 at=0,1.6,0` | SDF sphere |
| `capsule` | `capsule arm_L a=.2,1.4,0 b=.5,1.1,0 r=.05 r2=.04` | Tapered capsule between two points |
| `box` | `box base size=1,.1,1 at=0,0,0 round=.02` | Rounded box |
| `blend` | `blend body = torso head k=.05` | Smooth union (`k` = blend radius) |
| `cut` | `cut cup = outer inner k=.01` | Smooth subtraction |
| `inter` | `inter lens = a b k=.01` | Smooth intersection |
| `xform` | `xform head t=0,.05,0 r=0,15,0 s=1.1` | Move, rotate (degrees), scale |
| `mirror` | `mirror arm_L axis=x` | Creates `arm_R` |
| `mesh` | `mesh body res=128` | SDF → triangle mesh (marching cubes) |
| `del` | `del arm_R` | Delete |
| `cp` / `rollback` | `cp before_arms` / `rollback before_arms` | Named checkpoints |
| `undo` | `undo 2` | Undo the last N ops |
| `export` | `export body file=out/body.glb` | Export |

### Later commands (M5+), sketched
```
remesh body quad target=8k             # Blender QuadriFlow
style anime heads=7 build=slim sex=f age=adult
skel default_humanoid
prop shoulder=1.35h hip=1.45h waist=.95h leg_ratio=.52
sec thigh_L 0:(.62,.66) .5:(.48,.50) 1:(.34,.36)   # station:(width,depth) in head units
head template=anime_f eye=large jaw=narrow chin=.9
blend joints k=.04
mat skin toon ramp=2 shadow=#e8b4a8 rim=.3
rig auto                                # weights from the skeleton/section ownership
```

### Response format (example)
```
> sphere head r=.12 at=0,1.6,0
ok +head sdf bbox=.24x.24x.24
> blend body = torso head k=.05
ok +body sdf (torso,head consumed) bbox=.42x1.72x.31
> mesh body res=128
ok body mesh v=18.4k f=36.8k manifold=yes
```

---

## 6. MCP tool surface

| Tool | Input | Returns |
|---|---|---|
| `run` | DSL text (one or more lines) | One result line per op, plus errors |
| `inspect` | optional name | Compact scene tree, or details for one object |
| `measure` | name, optional region list | Dimensions, ratios in head units, volumes |
| `check` | name | Mesh health: manifold, holes, self-intersections, poly count, symmetry error |
| `view` | name, views=front,side,top,3q, res≤256, mode=sil\|shade | Small PNG(s). Use sparingly. |
| `compare` | name, ref image path, view | Silhouette IoU + per-region over/under % as text |
| `undo` / `checkpoint` / `rollback` | n or name | Result line |
| `export` | name, path, format=glb\|obj\|vrm | Path + stats |

Rule: every tool returns text first. Images only come from `view`.

---

## 7. Feedback module

- **measure:** bbox, height, part lengths, widths at named stations. Body context reports in head units (`height=7.02h shoulder=1.33h`).
- **check:** manifold, open edges, self-intersections, degenerate faces, poly count, L/R symmetry error.
- **silhouette:** orthographic silhouette rendered cheaply (numpy rasterization of the mesh, or Blender Workbench).
- **compare:** align a reference silhouette (e.g. from a ComfyUI turnaround sheet) by bbox, compute IoU, and report the worst regions in text: `thigh_L +8% wide, waist -5%`.

---

## 8. Tech stack

- **Python 3.11+** for the engine. The Blender worker uses Blender's bundled Python. If using the `bpy` wheel instead, match its required Python version.
- `numpy`, `scikit-image` (marching cubes), `trimesh`, `manifold3d`, `pillow`. Optionally `torch` for GPU SDF evaluation later.
- `mcp` Python SDK (FastMCP) for the server.
- `pytest` for tests.
- Blender 4.x+ headless for remeshing, rigging, materials, and rendering.

---

## 9. Milestones and acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| **M0** | Environment | `pytest` runs. Blender worker starts headless, creates a cube, exports GLB, and returns JSON. |
| **M1** | DSL + scene state | Parser handles all v0 syntax with line-numbered errors. Scene history, diffs, undo, and checkpoints work. Pure Python, fully unit-tested, no Blender. |
| **M2** | SDF core | All v0 primitives and smooth booleans. `mesh` produces a watertight mesh. `examples/snowman.form` (≤10 lines) exports a valid GLB. Same script gives the same hash. |
| **M3** | Feedback v1 | `measure`, `check`, and `view` (silhouette mode) work on the snowman. |
| **M4** | MCP server | Claude Code can build the snowman purely through MCP tools. Log the token use per session. |
| **M5** | Blender backend ops | `remesh quad` via QuadriFlow, smoothing, materials passthrough, GLB export through the worker. |
| **M6** | Skeleton + sections | `style`/`skel`/`prop`/`sec`/`mirror` produce a lofted, blended, quad-remeshed humanoid from ≤40 lines. |
| **M7** | Reference compare | `compare` against a front/side silhouette PNG gives per-region text feedback. The LLM can close a deliberate 10% proportion error in ≤3 iterations. |
| **M8** | Materials | `mat` with principled and toon presets, compiled to Blender shader nodes. |
| **M9** | Rigging + export | `rig auto` derives weights from skeleton ownership. GLB (and later VRM) loads in Godot and poses correctly. |

**Track token cost from M4 onward:** tokens per op, tokens per successful asset, and the iterations needed. These numbers decide what to add or simplify in the language.

---

## 10. Conventions for implementing sessions

- Only `blender_worker/worker.py` may import `bpy`. The engine must import and test without Blender.
- Every new DSL command needs: a parser test, an example in `examples/`, a one-line result format, and an error message with a fix hint.
- Keep responses terse. If a response exceeds ~150 tokens, summarize it and offer detail via `inspect`.
- Determinism: seed every random op, and use golden tests (measurements and hashes) for examples.
- Update this SPEC when the language changes.

---

## 11. Open questions

- Final name and file extension (`.form` for now).
- SDF evaluation on CPU (numpy) vs GPU (torch on the RunPod 5090) — start on CPU and switch once grids hit 256³+.
- Anime head: parametric SDF vs a deformed template mesh.
- Hair system (guide curves → cards/tubes): design it after M6.
- Licensing: Blender is GPL. Keep it behind the worker process boundary, and get proper advice before distributing anything commercially.
