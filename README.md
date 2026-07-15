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

Generate MVTec masks with HQ-SAM and bounding-box prompts:

```bash
PYTHONPATH=third_party/sam_hq python tools/generate_mvtec_foreground_sam.py \
  --root data/MVTec-AD --backend sam_hq --strategy v2 \
  --checkpoint checkpoints/sam_hq_vit_b.pth --categories all \
  --preview-dir runs/foreground-preview
```

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
