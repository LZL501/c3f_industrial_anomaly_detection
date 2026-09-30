# MVTec foreground audit beyond transistor

> Historical audit: the cable cut-face definition and ellipse correction below
> were rejected because they omitted visible outer sheath. The subsequent
> [direct per-image polygons](cable_manual_foreground.md) were also withdrawn
> because they included cast shadow. The current tool rejects those annotations.
> Use the [reviewed imagegen selection](cable_imagegen_foreground.md) for the
> current cable masks. Counts and checks below describe the earlier pass.

The second 2026-09-28 pass covers the remaining 14 categories and 3,416 normal
training images. Three systematic problems were repaired in 684 masks:

- **Metal nut:** previous hole filling marked the visible central void as
  foreground. Detect the aperture using the background intensity distribution,
  verify its location and dimensions, and retain the void as background.
  Preserve the existing outer silhouette and the visible metal bore wall.
- **Zipper:** the rectangular mask included white margins beside the woven
  tape. Follow the high-contrast fabric boundary row by row; retain the full
  fabric interior and teeth, including light markings inside the fabric.
- **Cable:** the fixed circle included parts of the dark rim/background beyond
  the visible cut face. Locate the bright-to-dark radial edge and fit a robust
  ellipse, rejecting isolated background texture responses. The intended ROI
  is the exposed face (cores and insulation), not the dark lateral surface.

The remaining object categories were retained after structural checks and
stratified visual review. Texture classes keep whole-image foreground: holes
in grid texture are part of its texture domain, unlike a nut's central void.

| Category | Images | Changed masks | Visual review samples |
| --- | ---: | ---: | ---: |
| bottle | 209 | 0 | 32 |
| cable | 224 | 224 | 42 |
| capsule | 219 | 0 | 33 |
| carpet | 280 | 0 | 8 |
| grid | 264 | 0 | 8 |
| hazelnut | 391 | 0 | 33 |
| leather | 245 | 0 | 8 |
| metal_nut | 220 | 220 | 53 |
| pill | 267 | 0 | 33 |
| screw | 320 | 0 | 34 |
| tile | 230 | 0 | 8 |
| toothbrush | 60 | 0 | 33 |
| wood | 247 | 0 | 8 |
| zipper | 240 | 240 | 45 |
| **Total** | **3,416** | **684** | **378** |

## Review coverage

All masks were checked programmatically for matching dimensions, binary values,
nonempty foreground, component counts, and category-specific geometry. The listed
378 samples were visually inspected using original-image overlays and contours
at 256x256; these comprise 338 object images and 40 texture images.
Selection includes 24 evenly spaced object images per category, extremes of
area/component count/border contact, and (for repaired categories) largest
changes and fitting outliers. Selected cable, nut, and zipper originals were
also inspected at source resolution.

This is full-dataset automated checking plus sampled visual inspection, not
individual visual acceptance of every image. Fine edge errors and unobserved
sample-specific defects remain possible. No independently labeled foreground
ground truth or accuracy improvement is claimed. No model training is involved.

## Reproduction

Run from the repository root with the original foreground masks available:

```bash
python tools/refine_mvtec_foregrounds.py \
  --data-root data/MVTec-AD --output-root artifacts/mvtec-refined

python -m pytest -q tests/test_mvtec_foreground_refinement.py \
  tests/test_foreground_annotations.py
```

The tool defaults to the 14 categories above; use `--categories` to select a
subset. It rejects a nonempty output root and does not replace the input masks.
Unchanged PNGs are copied byte-for-byte. The manifest records source, baseline,
and output hashes, pixel-change counts, fit details, and pending visual review.
Execution does not mark an image visually reviewed.

Outputs contain masks only. The saved experiment artifacts additionally include
a complete 15-category dataset view with original image/test symlinks and the
213 previously corrected transistor masks. Shared source masks are preserved.

## Checks and environment

Five new deterministic tests exercise nut apertures, slanted zipper boundaries,
border-touching masks, cable fitting with distracting background texture, and
texture/retained-mask behavior. The previous five transistor tests are also run.
The generated artifact report records full-data checks, independent mask
regeneration, and real-data pseudo-anomaly smoke results.

Offline CPU environment: Python 3.11.6, OpenCV 5.0.0, NumPy 2.4.6, SciPy 1.17.1,
Pillow 12.2.0, pytest 9.0.3, Ruff 0.16.9. PyTorch is not installed in this
environment. CUDA, Lightning, OmegaConf, and scikit-learn are not used by this
offline correction. Training code, losses, configurations, and evaluation
metrics were not modified.
