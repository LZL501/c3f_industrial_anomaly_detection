> WITHDRAWN: the user identified cast shadow included in the cable polygons.
> Do not treat this cable mask set as accepted. Visible sheath must remain
> foreground, but cast shadow on the background must be excluded.
> Passing format and pipeline tests did not establish boundary accuracy.

# Withdrawn cable polygon annotations

The former cable correction incorrectly kept only the pale cut face. The
foreground must include the visible outer sheath and lateral wall. That output
was rejected. The subsequent 224 individually traced polygons in
`assets/foreground/cable_manual.json` were also rejected because they included
cast shadow. The loader refuses these annotations. The details below describe
the withdrawn historical attempt; they are not instructions for producing
accepted masks. The replacement workflow is documented in
[cable imagegen foreground](cable_imagegen_foreground.md).

Each source image was inspected and its boundary vertices explicitly recorded.
The first eight references were inspected at source resolution; the remaining
216 were traced from 512-pixel views and their coordinates scaled by two.
Every final overlay was inspected at 256 pixels. Original-resolution overlays
and seven contact sheets are saved with the artifact. These are approximate
polygon annotations, not independently verified pixel-level ground truth.
Soft focus, cast shadows, and small edge details remain sources of uncertainty.

No color thresholds, fitted shapes, transferred contours, or segmentation
models determine the new cable boundaries. The renderer only validates the
source PNG hash and dimensions, fills the supplied polygon, and writes the
binary mask. Missing annotations fail rather than invoke an automatic fallback.

```bash
python tools/refine_cable_foregrounds.py \
  --data-root data/MVTec-AD \
  --output-root artifacts/cable-manual \
  --annotations assets/foreground/cable_manual.json

python -m pytest -q tests/test_cable_manual_foreground.py \
  tests/test_mvtec_foreground_refinement.py tests/test_foreground_annotations.py
```

The general refinement CLI also reads these manual cable annotations. Calling
its image-only `refine("cable", ...)` entry raises an error. The prior nut,
zipper, and transistor correction methods are unchanged; they should not be
described as a complete set of direct manual annotations.

All 224 masks are binary, nonempty, connected, match the source dimensions,
and exactly reproduce the supplied polygons. Connectivity is also checked
after the training loader's linear resize to 256 followed by a positive-value
threshold. Current tests cover missing annotations, source hash mismatches, and
rejection of the withdrawn annotation file. These checks validate data handling,
not boundary accuracy. The artifact also records texture/structure
pseudo-anomaly checks; no model training or accuracy comparison is claimed.

Runtime: Python 3.11.6, OpenCV 5.0.0, NumPy 2.4.6, SciPy 1.17.1,
Pillow 12.2.0, pytest 9.0.3, Ruff 0.16.9. PyTorch is unavailable;
CUDA, Lightning, OmegaConf, and scikit-learn were not used. Training behavior
and configs are unchanged. Masks and previews remain untracked artifacts.
