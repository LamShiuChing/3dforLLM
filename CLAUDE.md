# CLAUDE.md — formcode

formcode is a headless 3D modeling engine designed for LLMs to use: a terse DSL in, compact text feedback out, Blender as a backend worker. **Read `SPEC.md` before doing anything** — it holds the architecture, DSL, tool surface and milestones.

## How to work
- Work milestone by milestone (SPEC §9). Don't start a milestone until the previous one meets its acceptance criteria.
- Start at M0 unless told otherwise. Report which milestone you finished and what's next.
- Write tests alongside code; run `pytest` before declaring a milestone done.

## Hard rules
- Only `blender_worker/worker.py` imports `bpy`. The `formcode/` package must import and pass tests without Blender installed.
- The LLM-facing interface never exposes raw vertex lists.
- Every tool/op response is text, short (aim ≤150 tokens). Images only via `view`.
- All randomness is seeded; examples have golden tests.
- Every new DSL command ships with: parser test, example script, result-line format, error message with a fix hint.
- If you change the language or tool surface, update `SPEC.md` in the same change.

## Commands
- Tests: `pytest -q`
- Run an example (from M2): `python -m formcode run examples/snowman.form`
- Blender worker (from M0): `blender -b --python blender_worker/worker.py`
