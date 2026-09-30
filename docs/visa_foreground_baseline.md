# VisA foreground baseline

The official VisA images and annotations are downloaded from
[Amazon's dataset release](https://github.com/amazon-science/spot-diff).
The archive SHA-256 is
`2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`.
The split CSV and dataset license are pinned to official commit
`2a692ab575001cbde74d402d897a7286086c6199`.
The complete dataset contains 10821 images; the one-class split contains 8659
normal training images and 2162 test images. Only training images receive
foreground masks. Original anomaly annotations are preserved.

Two independent candidates are generated for every training image:

- Otsu: the category polarity and threshold from the original
  `ldm/data/visa2.py:MemSegDataset.generate_target_foreground_mask`. Capsules
  uses dark foreground; the other eleven categories use bright foreground.
  Extraction uses native image dimensions, without resizing or cleanup. This
  reproduces the old function for the same input pixels, rather than claiming
  equality with thresholding an already-resized training image.
- HQ-SAM: the existing ViT-B checkpoint and v2 pipeline, full-image box with
  a 3% inset, HQ output only, followed by the existing component cleanup.
  Masks retain source dimensions and multiple objects. Inference uses FP32.
  The full run uses eight L40S GPUs; twelve initial CPU outputs are reused.

Final selected masks are stored outside the dataset:

```text
visa-foreground/<category>/Data/Foreground/Normal/<image-stem>.png
```

The loader maps source `.JPG` names to these lossless PNGs when
`data.foreground_root` is set. The VisA configuration uses `visa-foreground`;
`tools/train.py --foreground-root` can override it. Configurations without a
separate root retain the original sibling-foreground lookup. Missing required
external masks raise an error instead of silently using a different mask.

The ignored artifact `artifacts/visa_foreground_baseline_20260929/` preserves
both candidate sets (`otsu_masks/`, `hq_masks/`), generation records, comparison
metrics, sample sheets, and explicit selection decisions. `STATUS.json` records
actual completion. `selected_manifest.json` identifies the source of every final
mask. None of these generated assets is committed.

Selection uses visual comparisons of evenly spaced and largest-disagreement
samples to choose a category default, with explicit per-image overrides where
reviewed. It is not exhaustive semantic inspection of all 8659 images. Pairwise
IoU measures agreement between methods, not foreground accuracy. Cast shadows,
transparent edges, thin pins, small holes, and background specks remain candidates
for later fine refinement. Both original candidates remain available. No mask
union, learned ranking, or manual repainting is applied during selection.

The initial selection visually compares 170 sample pairs across all 12 categories.
Category defaults use Otsu for chewinggum, fryum and pipe_fryum, and HQ-SAM for
the remaining nine categories. Reviewed chewinggum images 265 and 500 explicitly
use HQ-SAM because it preserves their shaded material better. The resulting
baseline selects 1351 Otsu masks and 7308 HQ-SAM masks. This selection does not
certify the unreviewed boundaries; both candidates remain available for refinement.

A subsequent review corrects 21 explicitly identified masks: 10 cashew, four
capsules, two chewinggum, three macaroni1, one pcb2 and one pcb4. Fourteen use
imagegen outputs, nine of which also receive individually traced boundary
corrections; seven use local point edits. Corrections remove cast shadows,
false holes and bridges, recover translucent shell edges, and restore verified
PCB through-holes. No source-color segmentation or automatic morphology is
used for these corrections. All generated attempts and previous masks are kept
in the ignored `artifacts/visa_foreground_refinement_20260929/` directory.

Its `current_manifest.json` records the current 8659 masks, and
`refinement_manifest.json` records the 21 replacements. All remaining 8638 masks
retain their baseline hashes. Both complete mask copies pass hash checks; the
21 corrected masks pass 42 real-data texture/structure anomaly-generation
checks. Each corrected capsule image contains 20 separate objects. This is
targeted refinement of known defects, not exhaustive manual dataset acceptance
or certification of pixel-perfect boundaries. The corrected masks use the same
external folder layout and are not included in the code repository.

Reproduce the candidate sets in new output directories:

```bash
python tools/generate_visa_foreground_legacy.py \
  --root data/VisA --output-root artifacts/visa-otsu \
  --report-dir artifacts/visa-otsu-report
PYTHONPATH=third_party/sam_hq python tools/generate_visa_foreground_sam.py \
  --root data/VisA --output-root artifacts/visa-hq \
  --report-dir artifacts/visa-hq-report \
  --checkpoint checkpoints/sam_hq_vit_b.pth --workers 1 --device cuda:0
```

Local Otsu runtime: Python 3.11.6, OpenCV 5.0.0, NumPy 2.4.6, SciPy 1.17.1,
Pillow 12.2.0, OmegaConf 2.3.1. The isolated loader-test/CPU-pilot environment
uses PyTorch 2.8.0+cpu and torchvision 0.23.0+cpu. GPU generation runtime:
Python 3.10.10, PyTorch 2.9.0+cu128, CUDA 12.8, torchvision 0.24.0,
OpenCV 4.13.0, NumPy 2.2.6, SciPy 1.15.3, Pillow 11.3.0, timm 1.0.16,
OmegaConf 2.3.0. Lightning and scikit-learn are not installed or used.
No model training or anomaly-detection accuracy comparison is performed.
