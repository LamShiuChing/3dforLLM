# Experiment 01 — dragon head "from nothing" (2026-10-12)

Setup: `lab.py` = numpy SDF vocabulary (Shadertoy/IQ idioms) + 4-view ortho renderer + text senses
(bbox per named part, floating pieces, symmetry, bound clipping). Model: Claude Opus 5.5 writing `dragon.py`.
Renders in `runs/v01..v05.png` (+ `_zoom`). No templates, no assets.

## Iterations
| v | change | program size | what I judged | judged by |
|---|---|---|---|---|
| 01 | blockout: cranium, snout, jaw, brows, horns, neck, eyes, nostrils | ~330 tok | reads as dragon/horse; front looks like a llama; no mouth | vision ✓ |
| 02 | wedge snout, mouth slit, teeth loop, jowls, cheek spikes, ridge loop | ~650 tok | "zipper" mouth, pipe neck; **couldn't tell if ridge spikes attached** | vision ✓ / ✗ |
| 03 | ridge re-placed *relative to neck surface*; S-neck; eyelids; nostril flares | ~1030 tok | text sense: ridge was floating 0.3 units above the neck in v02 → fixed, 0 floaters | **text** |
| 04 | voronoi scales, chin, throat plates, bust base | ~1280 tok | big jump in perceived detail; throat = "ribbed hose" | vision ✓ (needed zoom) |
| 05 | two scale sizes by region, subtler plates | ~1340 tok | recognizable, mid-detail dragon bust | vision |

Cost per iteration ≈ 0.4–1.3k tok of code edits + ~550 tok per 640px image + ~400 tok text report.
Whole loop ≈ 10k I/O tokens (excluding reasoning). Render 2.5–21 s on CPU.

## Findings
1. **Detail from programs works.** ~250 tokens of voronoi code produced thousands of scales; loops gave teeth/spikes/plates.
   The 150k-numbers-vs-100-decisions gap is bridged by code, not data.
2. **Vision is good at gestalt, bad at attachment.** Every "looks like X / reads wrong" call was right.
   It could not tell attached vs floating, or hole vs bump. Matches published results (VLMs miss floating parts).
3. **Text senses caught what vision missed.** The floating-ridge bug was invisible in renders, obvious in one text line.
   But a flag buried in the report (`clipped_at_bound`) was ignored by me → warnings must be top-line, not data.
4. **Relational placement fixes bugs permanently.** Anchoring spikes to the neck surface removed the bug class,
   but cost ~20 lines of vector math I had to write myself → this must be a protocol primitive
   (e.g. `on(neck, side=dorsal, t=.5)`), the single biggest missing piece.
5. **Seeing needs zoom.** At 320 px/view, surface detail is invisible; a zoom window was essential. The LLM must choose where to look.
6. **Senses need calibration.** Grid aliasing made spike tips look like floating parts (false positives cost tokens).
7. **Ceiling observed (not polish yet):** scales ignore form flow (uniform 3D voronoi, not aligned to surface direction),
   no secondary forms (muscle, skin folds), no hard edges, tube horns, box-cut mouth, no material/color,
   and no production steps (mesh, topology, UVs, rig) were attempted.

8. **The human reviewer caught two defects I didn't:** the eye sits "off" (a sphere pushed into a socket, not seated
   in the lids), and the teeth clip through the lower jaw. Both are *part-vs-part relations* (seating, interpenetration),
   the same blind spot as #2. → the engine should report interpenetration between named parts that aren't meant to blend.

# Experiment 02 — human body "from nothing"

`human.py`: skeleton first (named joints), flesh hung off joints, A-pose, 1.75 m. 4 iterations, ~1.6k-token program.
Mesh: `runs/human_v04.glb` — marching cubes at 384³, 91k verts, watertight, 1 body, 52 s.

- **Skeleton-first = zero floating parts** in every iteration (dragon v02 lesson applied by construction).
- **Text measurement beat vision on proportion:** report said 8.2 heads (too small a head); I hadn't seen it in the render. Fixed → 7.6.
- **Body converged fast** (mannequin quality by v02). Bead-like arms came from stacking muscle spheres — one tapered shape per segment reads better.
- **Face plateaued and regressed.** v03 "improvements" swallowed the lips and added a mask-like crease. Facial shapes are small and
  tightly coupled; each fix breaks a neighbour, and I can see "uncanny" but not *which primitive* causes it.
  → needs: revert/pairwise compare (research-backed), per-feature labeled close-ups, and probably a face-specific representation
  (landmarks + profile curves) rather than blended ellipsoids.
- Result quality: recognizable human, mannequin body, uncanny face. Far from production (no topology/UVs/rig; marching-cubes triangles).

# Experiment 03 — face with an anatomical vocabulary (same LLM, no models, pure numpy)

`facelab.py` adds: named landmarks + midline profile → RBF `fit` (surface passes through them), sculpt verbs along
surface-snapped curves (`crease/ridge/inflate/carve/groove`), almond-aperture `eye`, `ear`; senses: fit corrections per
landmark, landmark drift, profile deviation, labeled renders, side-by-side compare. `face.py` ≈ 1.46k tokens, 4 iterations.
Comparison: `runs/exp02_vs_exp03.png`. Mesh: `runs/face_v04.glb`.

| v | change | driven by |
|---|---|---|
| 01 | anthropometric landmarks + profile + 16 sculpt ops | LLM knowledge |
| 02 | base fixed per landmark (brow +10 mm, jaw −11 mm…); wide radius for side landmarks; `groove` op | **fit report** (text) |
| 03 | lower face pulled back 5–6 mm; fold widths ×2–3 (nasolabial 3→9 mm) | fit report + vision |
| 04 | wider midline bend, lower-lip mass, softer brow, thinner lid gap | vision (clean close-up) |

- **The thesis held: the vocabulary was the bottleneck.** Same model went from caricature (exp02) to a plausible, statue-like
  adult male head. The side profile was realistic on the *first* try — anthropometric knowledge in the LLM is good enough
  when it can express it as named points and a curve.
- **"Base was off by X mm at landmark" is the best sense so far.** It turned blockout error into a ranked to-do list; worst
  error went 11.4 → 4.3 mm in two iterations, each fix aimed correctly.
- **One LLM knowledge error:** fold widths were 3× too narrow (rendered as wire lines). Senses could flag this
  ("crease width 3 mm, typical nasolabial 8–10 mm") — an anatomy-scale lint.
- **Engine lessons:** small-radius fits make bumps (use wide radius for sparse landmarks); steep displacement breaks the SDF
  (slits must be true cuts: `groove`); thin lines read as drawn lines at normal zoom.
- **Still not realistic-production:** eyes read as cut slots, ears are paddles from the front, lips are simplified, no skin
  detail, no expression/muscle tension; muscles-on-body and fabric not tested yet.

### Exp 03 round 2 — fixing the user's 4 defects with general mechanisms (v05–v07)
User caught: ear not connected, eye too deep, nostrils too small, back of head distorted. Root causes and fixes:
| defect | root cause | mechanism added |
|---|---|---|
| ear | connected by a sliver (1 component, so "floating" check passed) | `ear()` snaps its root onto the head by ray cast; **seating sense** (% surface in contact) |
| eye deep | eyeball at absolute z; v02 moved the skin 5–10 mm forward around it | `eye()` takes (x, y) only; depth = skin found by ray cast − recess |
| nostrils | 4.5 mm (real ~11 mm): scale error | named ops + **scale lint** vs typical anatomical sizes |
| back of head | no back landmarks, and **no view ever showed the back**; front-fixed light made back views dark | 6 views incl. back, **camera-relative light**, back-midline landmarks |
Then the back view revealed a new bug: a sparse back profile (5 points) let the spline cut across the round skull → groove.
Fixed by following the skull curve. Nose wings still blobby; nostril/philtrum junction jagged (interlocking concave forms remain hardest).

## Protocol implications (to test next)
- Surface anchors as a primitive: `part.at(u,v)`, `dorsal/ventral/lateral`, `along(axis, t)`, `attach(a, to=b)`.
- Senses: warnings first (floating, clipped, intersecting, by part name); zoom; pairwise before/after; engine-answered questions.
- Surface-aligned detail: patterns that follow a flow field (scales, wrinkles, muscle fibers) instead of world-space noise.
