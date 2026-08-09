---
name: lifted-chip-mask-synthesis
description: "Generate a synthetic lifted-chip X-ray image and its exact ground-truth mask by procedurally painting a physically-motivated defect pattern onto a real normal (non-defective) chip crop, instead of extracting a mask from noisy real defective pixels. Use when creating labeled lifted-chip training data, tuning the visual realism of a painted defect (boundary shape, darkening profile, edge highlight) against real reference images, or explaining/debugging paint_synthetic_lifted_patch in vision_synthgen/lifted_chip_synthetic.py. ALWAYS ask whether a real good (non-defective) chip crop is available and whether the user's chip/defect matches this skill's supported visual model before generating anything."
argument-hint: '[optional] normal chip crop path or array; defaults to a synthetic uniform-gray demo chip'
user-invocable: true
disable-model-invocation: false
---

# Lifted Chip Mask Synthesis

Paint a synthetic lifted-chip defect onto a real normal chip crop and return an exact
pixel-level mask alongside it — no thresholding of real defective pixels required anywhere.

Implemented (currently a placeholder — not yet functional) in
`vision_synthgen/lifted_chip_synthetic.py` (`_wavy_half_mask`,
`paint_synthetic_lifted_patch`). Once implemented for real, author it in an nbdev
notebook (e.g. `nbs/10_lifted_chip_synthetic_gen.ipynb`) and export from there, per this
repo's nbdev convention — do not hand-edit the module directly at that point.

## Before Generating Anything — Two Required Gates

Run both of these *before* writing or running any generation code, every time this skill
is invoked:

**Gate 1 — Real good chip crop available?**
This skill only ever paints onto a real, defect-free chip crop — it never synthesizes chip
texture from scratch (see "Why Not Threshold Real Defective Images" below). Ask the user:
"Do you have a real GOOD (non-defective) chip crop — cropped to just the chip body, with any
die-edge/background margin removed — that I can paint the synthetic defect onto?" If no such
image exists yet, say so plainly and stop: offer to help them capture/crop one, but do not
fabricate a placeholder background and call the result realistic.

**Gate 2 — Does their described defect match this skill's supported pattern?**
Ask the user to describe (or show) their lifted-chip defect, then compare it against the
canonical pattern below ("What A Real Lifted-Chip Defect Looks Like"). Tell them explicitly
which parts match and which don't:
- If it matches (localized **darker** patch, wavy/curved boundary, gradient/brightness peaks
  right at that boundary and fades with distance on both sides, texture suppressed inside the
  dark region) → proceed with generation.
- If it doesn't match (e.g. their defect is uniformly bright rather than darkened, has no
  gradient peak at a boundary, or is actually a **void** — see the note below) → tell them
  plainly that there is no supported skill for that exact pattern yet, but offer to attempt a
  best-effort adaptation if they still want to proceed.

## What A Real Lifted-Chip Defect Looks Like (from reviewed reference images)

Reviewed against real reference X-ray crops, confirmed with the user:

- **Normal chip body**: textured — fine speckle grain plus a regular grid of small dot marks
  (e.g. solder-ball/via pattern).
- **Die-edge / background margin is NOT the defect.** Several reference crops include a smooth,
  lighter, wavy-bounded area with no dot-grid at all — that is simply *outside the chip* (die
  edge / background captured in the same crop), not a lifted-chip signature. Don't confuse this
  compositional artifact with the defect itself; a correctly cropped "good chip" reference should
  exclude it.
- **The lifted-chip defect itself is a localized DARKER patch inside the chip body** — the
  opposite of the die-edge margin above. It has a wavy/curved boundary, is darkest with a bright
  peaked rim right at that boundary, and both the darkening and the rim fade with distance from
  the boundary (this matches the original design in the Procedure section below almost exactly).
- **A bright, partial hook/"C"-shaped rim that only traces PART of a boundary (not the whole
  thing) is a VOID signature, not a lifted-chip signature** — out of scope for this skill. If a
  user's reference image shows that pattern, flag it as a void and say this skill doesn't cover
  it.
- **Texture is suppressed inside the darkened region**: the dot-grid/speckle pattern visible on
  normal chip fades out inside the darkened lifted patch, on top of the brightness valley itself
  (see Procedure step 4a below).

## When To Use

- You need labeled (image, mask) pairs for lifted-chip defects and don't have (or don't trust)
  pixel-thresholded masks from real defective images.
- You're tuning how realistic a painted defect looks against a real reference image (boundary
  shape, how dark it is, how far the darkness penetrates, whether it has a bright edge).
- You're extending this pattern to a new/related defect type and want the same procedural,
  no-thresholding approach.
- You're debugging or explaining `paint_synthetic_lifted_patch`'s behavior.

## Why Not Threshold Real Defective Images Instead

Pixel-level thresholding (Otsu, `mean - k*std`) on real lifted-chip crops is unreliable as a mask
source: real per-chip material grain/texture makes the outcome depend on how that texture happens
to sit relative to the computed threshold — clean blob on some chips, salt-and-pepper speckle on
others. Painting the defect yourself onto a real *normal* chip sidesteps this: the mask is exact by
construction, because you drew it.

## Procedure

1. **Shape the boundary, not a blob.** A real lifted chip is tilted about one edge — one side is
   darker than the other, separated by a *curved* boundary — not a random floating circle.
   `_wavy_half_mask` builds this as a signed-distance half-plane (`angle_deg` = tilt direction)
   plus a sinusoidal perturbation (`wave_amp_frac`, `wave_freq`) so every parameter has a direct
   physical meaning and stays reproducible with plain trigonometry.
2. **Target an area fraction, not a fixed shape.** Because the wavy boundary + clipping to the
   chip's segmentation footprint make the true area-vs-offset relationship non-trivial, search a
   small range of boundary offsets (~25 candidates, pure numpy, no image I/O) and keep whichever
   lands closest to the requested `area_frac_range`.
3. **Compute signed distance from the boundary**, not just inside/outside: positive and increasing
   with depth into the lifted-side mask, negative and decreasing with depth into the normal side,
   exactly 0 at the boundary (`cv2.distanceTransform` on the mask and its inverse).
4. **Darken as a valley in that signed-distance space, not a constant offset across the whole
   masked region.** A real lifted chip stays close to its natural brightness almost everywhere and
   only darkens near the boundary, blending back toward baseline with distance — a uniform
   half-region offset reads as an artificial flat block once compared against a real image.
   `dark_penetration_frac_range` controls how far the valley persists before fading:
   - Low (~0.15): thin dark band right at the boundary, rest of the "lifted" side stays light
     (matches a mild/early-stage lift).
   - High (~0.7–1.0): darkness persists across most of the region (matches a severe lift with a
     genuine broad dark half — measured with `lift_dark_ratio` up to 0.5–0.6 against real chips in
     the source repo this skill was ported from; re-verify against this repo's own reference
     images once a gradient-measurement notebook exists here).
   - Both regimes are real; randomize per sample rather than picking one universal shape.
4a. **Suppress the chip's normal texture inside the darkened region, don't just darken it.** A
    real normal chip crop typically shows fine speckle grain plus a regular grid of dot marks
    (solder-ball/via pattern); confirmed against reference images that this texture visibly fades
    inside the lifted-dark patch, not just its overall brightness. Blend toward a locally-smoothed
    (e.g. small-kernel Gaussian-blurred) version of the source crop, weighted by the same
    signed-distance valley used for darkening in step 4, rather than only subtracting/scaling
    intensity — darkening without blurring still shows the untouched dot-grid through the dark
    patch, which reads as fake.
5. **Add a bright edge ridge exactly at the boundary, as a true peak — not a flat plateau.** Real
   lifted chips show a thin bright rim (X-ray edge-enhancement / bond-wire-projection signature at
   the physical step discontinuity), which is what gives them a convincing "3D" look. Build it as
   `exp(-(dist_from_boundary**2) / (2*sigma**2))` from the same signed-distance field — peaked
   exactly at the boundary, decaying symmetrically on both sides. **Do not** build this as
   "dilate the boundary, then blur" — that produces a flat-topped plateau (uniform brightness
   across a widened stripe, only its two outer edges soft), which is visibly wrong compared to a
   real example. Do not confuse this with the partial bright hook/"C"-shaped rim seen in some
   reference images — that traces only PART of a boundary and is a **void** signature, out of
   scope here (see "What A Real Lifted-Chip Defect Looks Like" above).
6. **Return `(painted_image, mask)`** where `mask` marks the entire lifted-side region as
   foreground `1` (same convention as `build_lifted_region_library`'s `lifted/*/mask.png`), even
   though the visual darkening may only occupy part of it near the boundary — the mask is a
   *label* for where the physical lift is, the paint is a *visual approximation*, and the two are
   intentionally decoupled.

## Validation Checklist

- **Area check**: `mask.mean()` lands near the requested `area_frac_range`.
- **Blends-back check** (low `dark_penetration_frac`): near-boundary pixels are measurably darker
  than baseline; pixels deep inside the masked region (far from the boundary) are close to
  baseline — confirms the valley fades, doesn't persist forever.
- **Stays-dark check** (high `dark_penetration_frac`): deep-interior pixels remain notably darker
  than baseline — confirms the broad-dark-half regime still works.
- **Peaked-ridge check**: sample intensities along a line perpendicular to the boundary; they must
  strictly decrease moving outward (a flat plateau for several pixels means the ridge is built
  wrong — see step 5).
- **Texture-suppression check** (step 4a): sample local variance/std in a small window inside the
  darkened region vs. an equivalent window on untouched normal chip — variance should drop inside
  the dark patch, not just mean intensity; an unchanged dot-grid still visible through the dark
  patch means step 4a wasn't applied.
- **Side-by-side visual QA**: render the painted chip next to a real reference lifted-chip image
  (same colormap) and look for mismatches — e.g. "this looks like a uniform stripe", "the real one
  has a strong edge / 3D feeling", "the dark part should blend back to light with distance from
  the boundary". Each of these observations previously caught a real bug in the model (uniform
  half-darkening, flat-plateau ridge) — trust visual mismatches over the code looking "reasonable".
- **Quantitative QA**: compare synthetic dark-region intensity ratios against a real population of
  measured `lift_dark_ratio` values once such a measurement notebook exists in this repo (ported
  from the source repo's `compare_synthetic_vs_real_lift_ratios` / `compute_dark_blob_features`) —
  confirm synthetic ratios actually overlap the real population's range, not just look visually
  similar.

## Common Pitfalls (already hit once each — don't repeat)

- **`cv2.seamlessClone` (Poisson blending) washes out smooth/large-scale darkness.** It only
  preserves the source's local Laplacian (texture/edges), not a flat offset — pasting a uniform
  dark region onto a uniform background via `NORMAL_CLONE`/`MIXED_CLONE` returns a result close to
  the *destination's* level, not the source's. Use direct feathered-alpha blending for smooth
  defects instead (fine for voids/sharp-edged blobs, which do have real local texture).
- **Mask value convention mismatches (0/1 vs 0/255) cause silent `uint8` overflow** in alpha-blend
  math (`255 * 255 mod 256 = 1`). Always normalize a loaded-from-disk mask to strict 0/1
  immediately, before any further arithmetic.
- **A uniform half-region darkening looks artificial** — build the darkening as a distance-from-
  boundary profile (this skill's step 4), not a constant offset.
- **"Dilate then blur" for a highlight ridge produces a flat plateau, not a peak** — use a distance
  transform + Gaussian instead (step 5).
- **Confusing the die-edge/background margin in a raw crop for the lifted-chip defect itself.**
  Some raw reference crops include a smooth, dot-grid-free area that is just the die edge /
  background caught in frame — not a lift. A "good chip" reference must exclude this margin;
  don't paint a defect there or measure baseline statistics from it.
- **Confusing a partial bright hook/"C"-shaped rim for the lifted-chip boundary ridge.** That
  pattern is a **void** signature (tracing only part of a boundary), out of scope for this skill —
  don't try to reproduce it here.
- **Skipping the two pre-generation gates** (real good-chip crop available? user's defect actually
  matches this pattern?) and generating anyway — leads to confidently wrong output when the user's
  actual defect is a different type (e.g. a void) or no real reference exists yet.

## Example Prompts

- "Generate a synthetic lifted chip and mask from this normal chip crop." → first confirm the crop
  is a real, defect-free chip-only crop (Gate 1), then confirm the described defect matches the
  canonical pattern above (Gate 2), then generate.
- "Compare a painted lifted-chip patch against this real reference image — does the darkening
  profile look right?"
- "Add a new penetration/severity preset to paint_synthetic_lifted_patch for [some scenario]."
- "Here's my lifted chip image, can you make me a synthetic version + mask?" → ask Gate 1 and Gate
  2 questions first; if the image actually shows a void (partial bright rim) or has no real good
  reference crop, say so plainly before attempting anything.
