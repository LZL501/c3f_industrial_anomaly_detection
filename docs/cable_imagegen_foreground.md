# Cable foreground with imagegen

Current scope: bulk generation stopped at the user's request. All 179 already
generated masks are used after review and local corrections; the remaining
45 cable masks are retained byte-for-byte from the original shared training
foregrounds. They were not re-reviewed in this pass. Use `export_selected.py`;
the former full-batch export is disabled.

The target foreground includes the physical outer sheath and visible lateral
wall, and excludes cast shadow on the background. Both preceding cable exports
were withdrawn: the ellipse omitted sheath; the manually traced polygons
included cast shadow.

Each cable photograph is passed separately to the built-in imagegen editor to
produce an aligned white silhouette on black. Raw outputs are retained. The
normalizer decodes the generated labels at 128 and maps the canvas back to the
source dimensions with nearest-neighbor resize. It does not segment source
colors, fit shapes, transfer contours, or smooth boundaries.

Every resulting mask is inspected as a 512-pixel overlay on its source.
Questionable small features are inspected at source resolution. Explicit local
pixel-coordinate patches correct confirmed errors; source-supported fibers and
material irregularities are retained. Fine fibers and soft edges remain
uncertain. These masks are reviewed approximations, not independent ground truth.

The ignored artifact `artifacts/foreground_cable_imagegen_20260928/` contains
raw images, prompts, normalized masks, per-image provenance and review records,
local point patches, overlay previews, and validation/export scripts.
`STATUS.json` records the actual completion state; `REPORT.md` and
`validation.json` record validation of the 179 generated masks and provenance
of the 45 retained masks.

The export builds a separate dataset view with new cable masks and links to the
3405 unchanged masks in the other 14 categories. Original shared data remains
untouched. Zipper retains its existing foreground masks. That combined export was used before the separate foreground directory
described below.

Validation checks binary dimensions, source/output hashes, exact reproduction
from saved generated masks and recorded patches, reviewed topology, and texture
and structure pseudo-anomaly samples confined to foreground. These checks do
not establish boundary accuracy. No model training or AUROC claim is made.

Runtime: Python 3.11.6, OpenCV 5.0.0, NumPy 2.4.6, SciPy 1.17.1,
Pillow 12.2.0, pytest 9.0.3, Ruff 0.16.9. PyTorch is unavailable;
CUDA, Lightning, OmegaConf, and scikit-learn are not used by this offline workflow.

Current local layout: final masks are consolidated in
`mvtec-foreground/<category>/train/foreground/<image>.png`, separate from the
original dataset. `data/MVTec-AD` links to the original shared dataset;
`configs/mvtec.yaml` uses `data.foreground_root: mvtec-foreground`.
The older combined export remains a historical artifact. The local folder
contains 3629 byte-verified masks, including all 179 generated cable masks and
45 retained cable originals. Provenance is outside the mask directory at
`artifacts/mvtec_foreground_layout.json`. No masks were regenerated for this
reorganization. The shared filesystem did not respond during this operation;
the separate folder was built from the local files verified against their
saved export manifests. It has not been copied to a new CephFS location.

Runtime for the separate-root loader change: Python 3.11.6, NumPy 2.4.6,
OpenCV 5.0.0, SciPy 1.17.1, OmegaConf 2.3.1. PyTorch, torchvision,
Lightning and scikit-learn are unavailable; CUDA and training were not run.

Separate-root checks: all 3629 hashes and source filename sets matched; four
offline checks using the actual foreground/factory functions passed, including
external mask precedence and missing-mask failure. Changed-file Ruff lint and
format checks passed. Full data test collection was blocked by missing PyTorch;
no model or training smoke test was run.
