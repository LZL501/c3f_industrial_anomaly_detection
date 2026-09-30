# C3F Clean

A focused PyTorch implementation of C3FGS, extracted from the
`model_for_visa2.py` path in the original repository. It keeps the C3F
reconstruction, pseudo-anomaly training, guided segmentation, evaluation, and
visual logging paths while removing the unused latent-diffusion stack and the
PyTorch Lightning dependency.

## Implementation

- Frozen ImageNet `IMAGENET1K_V1` Wide-ResNet50-2 features at four stages.
- Coreset memory initialized from normal support images with folds
  `[8, 4, 2, 1]`.
- Coarse-to-fine fusion with `max(cos(f, q), 0)` and a three-stage averaged
  rough anomaly map.
- Six-channel `[input, reconstruction] * rough_map` guided segmentation.
- L1, LPIPS, feature, memory, adaptive adversarial, and focal losses matching
  the released training path.
- DTD and structural pseudo anomalies restricted to validated foreground masks.
- PyTorch checkpoints, JSONL metrics, and TensorBoard scalar/image logging.

The default `train.freeze_codebook: true` matches the released code. Set it to
`false` to train memory embeddings as described in the paper.

## Install

Use Python 3.10 or newer and install a CUDA-compatible PyTorch build first when
training on GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download DTD to the default `data/dtd/images` path:

```bash
bash scripts/download_dtd.sh
```

## Foreground Masks

Object categories require one foreground mask per normal training image. Masks
use the dataset layout `<category>/train/foreground/<image>.png`; texture
categories use the full image. Training fails on missing, unreadable, or empty
masks so pseudo anomalies cannot silently spill into the background.

The current MVTec masks are stored separately under
`mvtec-foreground/<category>/train/foreground/<image>.png`. Set
`data.foreground_root` (or `tools/train.py --foreground-root`) to this folder;
`data.root` points to the original image dataset. The MVTec template uses this
separate mask directory. If `foreground_root` is omitted, older configurations
continue reading masks beside the training images. Required external masks
must exist; a missing external mask does not fall back to a dataset mask.

VisA uses the separate `visa-foreground/<category>/Data/Foreground/Normal/`
folder with PNG masks matching each source image stem. Its Otsu and HQ-SAM
candidates and selection limits are described in
[VisA foreground baseline](docs/visa_foreground_baseline.md).

Generate MVTec masks with HQ-SAM and bounding-box prompts:

```bash
PYTHONPATH=third_party/sam_hq python tools/generate_mvtec_foreground_sam.py \
  --root data/MVTec-AD --backend sam_hq --strategy v2 \
  --checkpoint checkpoints/sam_hq_vit_b.pth --categories all \
  --preview-dir runs/foreground-preview
```

For transistor, refine those initial masks using the reviewed training-image
annotations. The original geometric masks can omit package edges and include
background beside the leads. Write corrections to a separate mask root:

```bash
python tools/refine_foreground_from_annotations.py \
  --data-root data/MVTec-AD --output-root artifacts/transistor-refined
```

This writes masks and a provenance manifest, not a complete dataset copy.
Review the overlays before using the masks for training. See
[foreground refinement](docs/foreground_refinement.md) for the annotation scope,
reproduction checks, and remaining limits.

Audit and refine the non-cable MVTec categories with:

```bash
python tools/refine_mvtec_foregrounds.py \
  --data-root data/MVTec-AD --output-root artifacts/mvtec-refined \
  --categories bottle capsule carpet grid hazelnut leather metal_nut \
    pill screw tile toothbrush wood zipper
```

The cable ellipse and manual-polygon exports were both withdrawn: one omitted
sheath, the other included cast shadow. The current replacement uses imagegen
with per-image overlay review: 179 generated masks are included, while the
remaining 45 cable masks retain their original shared versions. See
[cable imagegen foreground](docs/cable_imagegen_foreground.md).
The command above excludes cable; the tool rejects the withdrawn cable
annotations. It still repairs metal-nut apertures and
zipper fabric edges, preserves other object masks, and uses full-image texture
foreground. Transistor uses the separate annotation tool above. See the
[historical audit](docs/mvtec_foreground_audit.md) for that pass's review scope.

## Train And Evaluate

```bash
python tools/train.py --config configs/mvtec.yaml \
  --data-root data/MVTec-AD --texture-root data/dtd/images \
  --category bottle --device cuda:0

python tools/eval.py --config configs/mvtec.yaml \
  --checkpoint runs/mvtec_c3f_bottle/best.pth \
  --data-root data/MVTec-AD --category bottle --device cuda:0
```

Resume all model and optimizer state with `--resume runs/.../last.pth`. Override
configuration values using dotted arguments such as `train.epochs=100`.

Training writes `metrics.jsonl`, `last.pth`, `best.pth`, and TensorBoard events
under `runs/<experiment>_<category>/`. Inspect logs with:

```bash
tensorboard --logdir runs
```

Run focused checks with `pytest -q`.
