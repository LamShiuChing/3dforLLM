# CLAUDE.md — formcode

formcode lets frozen LLMs build detailed 3D from nothing: a small Python SDF vocabulary in, compact text senses out.
**Read `SPEC.md` first** (architecture, scene contract, vocabulary, milestones P0–P6). `HANDOFF.md` has the current state and next step.

## Non-negotiables (from the user)
- The LLM is the creator: no 3D-generative models, no shape templates or libraries. Image models only as *input* (reference sheets).
- Open source (Apache-2.0), lightweight: core = numpy + scipy + pillow. GPU optional.
- Borrow existing languages and idioms rather than invent syntax.

## How to work
- Work milestone by milestone (SPEC §9). Each one ends with an experiment and a measurement. Report which milestone you finished and what's next.
- Show results: render, look at all views (including back/top), run every sense. The user catches visual defects; be blunt about what's still wrong.
- Write tests alongside code; run `pytest -q` before declaring a milestone done.

## Hard rules
- The `formcode/` package never imports Blender (`bpy`). Blender lives behind a process boundary in `later/` (P6).
- The LLM-facing interface never exposes raw vertex lists.
- Every sense runs on every report; a new sense is on by default.
- All randomness is seeded; examples have golden tests. Regenerate goldens (`python tests/test_examples.py`) only for intended changes, and say so.
- `experiments/` is the frozen session-1 record; don't edit it, port from it.
- If you change the vocabulary, scene contract, report format or CLI, update `SPEC.md` in the same change.

## Commands (Git Bash, venv in `.venv`, package installed editable)
- Tests: `.venv/Scripts/python.exe -m pytest -q` (~35 s)
- Render + report: `.venv/Scripts/python.exe -m formcode examples/face.py out.png [cx,cy,cz,half]`
- Mesh: `.venv/Scripts/python.exe -m formcode examples/face.py out.glb [res]`
- Windows: cp950 code page, so open files with `encoding="utf-8"` and set `PYTHONIOENCODING=utf-8` for unicode output.
