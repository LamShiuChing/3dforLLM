# What the experiments taught us (2026-10-12)

Three experiments, one LLM (Claude Opus 5.5), no templates, no generative models, pure numpy:
- **exp01 dragon head**: SDF primitives + smooth blends, 5 iterations.
- **exp02 human body**: skeleton-first SDF, 4 iterations.
- **exp03 face**: anatomical vocabulary (landmarks, profile curves, sculpt verbs), 7 iterations.

Details: `sdf_lab/FINDINGS.md`. Renders and meshes: `sdf_lab/runs/`.

## Findings

1. **Short programs can carry detail.** About 250 tokens of Voronoi code made thousands of scales, and loops made teeth, spikes and plates. Detail should come from code, not data.
2. **The vocabulary is the realism bottleneck, not the LLM's knowledge.**
   - With blended ellipsoids, the same model produced a caricature face.
   - With landmarks, profile curves and sculpt verbs, it produced a plausible human head.
   - The profile was realistic on the first try.
3. **Text senses beat vision for relations and measurements.** Examples:
   - floating spikes (invisible in the renders)
   - head-unit proportion (8.2 heads → 7.6)
   - "base was X mm off at landmark Y", the best sense so far: the worst error went 11 → 4 mm in two iterations.
4. **The LLM's vision is good at overall impressions and bad at:**
   - how parts attach and sit
   - telling a hole from a bump
   - anything not in view (it never looked at the back of the head)
   - seeing its own mistakes: it sees what it meant to build.

   The human caught about 10 defects the model missed: the dragon's eye and teeth, plus the ear, eye depth, nostrils and back of the head in exp03.
5. **Absolute coordinates cause regressions; relative placement removes whole bug classes.**
   - skeleton-first meant zero floating parts
   - spikes anchored to the neck surface
   - eye depth taken from the skin
   - ear snapped onto the head
6. **The LLM knows *where*, not *how big*.** Folds came out 3× too narrow and nostrils 2× too small. A scale lint against typical sizes catches this.
7. **The hardest shapes are concave and interlocking:** nostrils, ear folds, eyelids around the ball, mouth corners. These are still the weakest results.
8. **Engine math traps must be impossible by construction.** The LLM shouldn't need to know about any of these:
   - steep displacements break the distance field
   - small-radius landmark fits make bumps
   - sparse splines cut corners
   - a fixed light leaves the back of the head dark
9. **Cost:**
   - scene programs are about 1.3–1.6k tokens
   - a session is about 10–20k I/O tokens per asset, excluding reasoning
   - renders take 10–25 s on CPU
10. **Not tested yet:**
    - muscles on the body
    - fabric
    - hard-surface objects
    - production output (clean topology, UVs, rig)

## exp04 — anime head from a reference sheet (`anime_ref/`, 8 iterations)
The reference sheet was a front and side view generated with Anima in the local ComfyUI. Landmarks were read off grid overlays, and a new sense compares the model's outline with the reference per band, in mm.

- **A reference plus a text comparison converges fast.** Every band in both views reached about 1–2 mm average error within a few iterations, and each "move X by N mm" correction landed.
- **The engine measures the reference better than the LLM reads it.** My grid readings were off by up to 12 px (≈7 mm). So the engine should extract outlines automatically, and the LLM should only label meaning (eye corners, mouth).
- **Senses must know what they measure.** False errors came from:
  - hair in the outline (fixed with a skin mask)
  - pink hair highlights (fixed with a hue rule)
  - the neck behind the chin in the front silhouette
  - a contact-% seating metric applied to ears, which are *supposed* to stick out (the junction thickness is what matters)
- **Matching two silhouettes ≠ good 3D (Goodhart).** At about 1 mm outline error the head still had:
  - floating ears
  - a forehead ridge
  - flat side bands left by the trim

  Only the always-on full report and the extra views exposed these. Every check must run on every iteration.
- **Visual hull (trimming to a 2D outline)** removes depth guessing and matches the silhouette by construction, but hard trimming leaves flat walls. It should be applied softly and locally.
- **Swing-around-root (`splay`) keeps attachments**; a sideways shift (`flare`) detaches them. Articulated parameters beat offsets.
- **Stylized looks live largely in texture:** painted eyes, brows and hair. Untextured grey geometry of an anime head looks alien even when the shape matches. Anime needs eye/brow materials and a hair system, not only shape.

## Published research agrees (agent reports, sources in the conversation)
- Text descriptions of space beat images for frontier models (SpatialEval). Orientation judged from pixels scores 23–37% (3DSRBench).
- Open-ended self-critique on renders barely helps. Floating or disconnected parts are the most common failure in LLM 3D-by-code (3DCodeBench, CADCodeVerify).
- What works:
  - pairwise before/after comparison with a revert (BlenderAlchemy)
  - stating relations and letting a solver compute the numbers (SceneCraft, Holodeck)
  - pointing at parts by label (Set-of-Mark)
- LLM sculpting is almost unexplored. ViSculpt (2026) drives the Blender GUI with simulated mouse input. We found no system where an LLM issues text strokes on anatomical curves.
