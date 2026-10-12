# HANDOFF — read this first (state as of 2026-10-12, end of session 1)

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
Session 1 was **research + experiments**. No engine code yet.
- `experiments/SUMMARY.md` has the findings across 4 experiments. **Read this.**
- `PLAN.md` has the draft engine plan, milestones P0–P6 and 4 open decisions.
- `experiments/sdf_lab/FINDINGS.md` is the detailed per-iteration log.
- `SPEC.md` / the original `CLAUDE.md` text are the **outdated Blender-first design**. `PLAN.md` supersedes them; a rewrite is pending.
- `blender_worker/`, `formcode/backend/client.py`, `tests/test_worker.py` hold early M0 scaffolding. Planned to move to
  `later/` for production steps (topology/UV/rig). Its pytest has not been run in the project venv.

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

## Open decisions (the user must confirm before P0). My proposed defaults:
1. Python vocabulary (not a custom DSL).
2. Rewrite `SPEC.md` + `CLAUDE.md` around `PLAN.md`.
3. Apache-2.0 license; keep the name `formcode` for now.
4. Move the Blender M0 files to `later/`.

**Next session starts with:** confirm the decisions → rewrite the spec → **P0**: turn `experiments/sdf_lab` into a `formcode`
core package (kernel + renderer + always-on senses) with tests that reproduce the experiments. The experiments then become benchmarks
that track tokens, iterations and defects.

## How to run things (Windows 11, Git Bash)
- venv: `.venv` (Python 3.12, numpy, pillow, scipy, scikit-image, trimesh, pytest), created with `uv venv --python 3.12`.
- Lab render: `cd experiments/sdf_lab && ../../.venv/Scripts/python.exe lab.py dragon.py out.png [cx,cy,cz,half]`.
  An output path ending in `.glb` exports a mesh.
- Face lab: `../../.venv/Scripts/python.exe facelab.py face.py out.png [cx,cy,cz,half]`. Prints the senses report.
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
