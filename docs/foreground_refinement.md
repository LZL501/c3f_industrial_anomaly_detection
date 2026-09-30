# Transistor foreground refinement

The original HQ-SAM/geometry pipeline removes circular board-hole regions even
where they overlap the package and fills leads with fixed-width polylines. This
can omit real package pixels and include nearby background. The new offline
refinement tool repairs these masks without changing the training algorithm.

## Annotation and generation

`assets/foreground/transistor_manual.json` contains visible-silhouette polygons
traced against 15 original 1024x1024 MVTec transistor training images.
Each record has the source PNG SHA-256 and separate body/three-lead polygons.
Dark copper terminal tips are included; board holes and cast shadows are not.
These are approximate visual annotations, not independently labeled ground truth.

Eleven images act as exemplars. Four additional images (`061`, `121`, `143`,
`152`) use direct manual overrides after visual review found transfer failures;
they are never used as exemplars. For the remaining images, three nearby
exemplars are warped into the query image, followed by a narrow boundary
adjustment. Image gradients near the baseline package locate its four edges,
avoiding the hole-shaped notches and false board-hole protrusions.

The 2026-09-28 output covers all 213 transistor training images: 15 direct
annotations and 198 transferred/refined masks. All 213 contour overlays were
inspected at 256x256 and the 15 annotated originals at source resolution. This
is a coarse silhouette review, not a claim of pixel-perfect edges. Other
MVTec categories and VisA were not re-annotated in this pass.

## Reproduce

Run from the repository root, with the original foreground masks still in the
input dataset's `transistor/train/foreground` directory:

```bash
python tools/refine_foreground_from_annotations.py \
  --data-root data/MVTec-AD --output-root artifacts/transistor-refined

python tools/review_mvtec_foreground.py \
  --data-root data/MVTec-AD --mask-root artifacts/transistor-refined \
  --review-dir artifacts/transistor-review --categories transistor --category-sheets

python -m pytest -q tests/test_foreground_annotations.py
```

The output root must be separate and its foreground directory empty. The tool
verifies reference image hashes and dimensions, saves 0/255 PNG masks, and
records input/output hashes and annotation provenance. It deliberately records
visual review as pending; execution alone is not an inspection. To evaluate
transfer without directly copying a query's annotation, pass `--leave-one-out`
with another output root. The four override-only images remain excluded from
the exemplar pool.

Generated outputs are ignored by Git. Keep source images, baseline masks, new
masks, review pages, and the manifest together as experiment artifacts. The
mask-only output is not a complete training dataset; use a separate dataset
view containing these masks and the original images/test data.

## Validation and limits

- All 213 output PNGs reproduced byte-for-byte with the saved generation tool.
- Five deterministic tests cover overlapping polygon roots, disconnected leads,
  mismatched source images, retained dark terminal tips, and package repair.
- All 213 masks remain connected at original resolution and after the training
  loader's 256x256 resize/threshold operation. A seeded smoke check generated
  1,704 pseudo anomalies (850 DTD, 854 structural): none changed background
  pixels or placed anomaly labels outside the resized foreground.
- Against the 11 initial visual references, mean mask IoU changes from 0.8869
  for the old masks to 0.9549 for automatic refinement with the query excluded
  from exemplar selection. This is a development-set diagnostic: these images
  guided implementation choices. It is not an independent benchmark and is
  not anomaly-detection AUROC. Final masks for these 11 use the direct polygons.
- No model training or anomaly-detection evaluation was run for this correction.
  Improved silhouettes do not establish improved model accuracy.

The tool was checked on CPU with Python 3.11.6, OpenCV 5.0.0, NumPy 2.4.6,
Pillow 12.2.0, SciPy 1.17.1, and pytest 9.0.3. PyTorch/CUDA/Lightning/OmegaConf/scikit-learn
were not used by this offline tool; the full model test suite requires a
separate training environment. Exact smoke-test runtime versions and outputs
are recorded with the generated artifacts.
