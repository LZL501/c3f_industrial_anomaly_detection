# Foreground preparation

Training uses foreground masks to keep synthetic anomalies on the object. Store masks outside the original datasets, using the layouts in the [README](../README.md#2-prepare-data-and-foreground-masks). The tools below generate **candidates**; compare their overlays with the source images before selecting masks for training.

## Optional HQ-SAM dependency

Follow the [official HQ-SAM installation instructions](https://github.com/SysCV/sam-hq#standard-installation). The scripts in this repository use the source checkout's `segment_anything` interface:

```bash
git clone https://github.com/SysCV/sam-hq.git artifacts/sam-hq
pip install -e artifacts/sam-hq
pip install timm
```

Download the **HQ-SAM ViT-B** checkpoint from the upstream repository's [model checkpoints](https://github.com/SysCV/sam-hq#model-checkpoints) and place it at `checkpoints/sam_hq_vit_b.pth`. The checkpoint and HQ-SAM source are not bundled here.

## MVTec-AD candidates

Generate into a separate candidate directory:

```bash
PYTHONPATH=artifacts/sam-hq python tools/generate_mvtec_foreground_sam.py \
  --root data/MVTec-AD --output-root artifacts/mvtec-hq-candidates \
  --backend sam_hq --strategy v2 --model-type vit_b \
  --checkpoint checkpoints/sam_hq_vit_b.pth --categories all \
  --preview-dir artifacts/mvtec-hq-preview --device cuda:0
```

The mask path is `<category>/train/foreground/<image-stem>.png`. After review, place the accepted masks under `mvtec-foreground/` with the same relative paths. Texture categories use the entire image; zipper still requires its object foreground.

For the reviewed transistor annotations and their reproduction workflow, see [transistor refinement](foreground_refinement.md). For the current cable selection, see [cable foreground](cable_imagegen_foreground.md). The withdrawn cable polygon annotations remain historical records and are rejected by the refinement loader.

## VisA candidates

Use the official one-class split at `data/VisA/split_csv/1cls.csv`. Generate Otsu and HQ-SAM candidates separately:

```bash
python tools/generate_visa_foreground_legacy.py \
  --root data/VisA --output-root artifacts/visa-otsu-candidates \
  --report-dir artifacts/visa-otsu-report

PYTHONPATH=artifacts/sam-hq python tools/generate_visa_foreground_sam.py \
  --root data/VisA --output-root artifacts/visa-hq-candidates \
  --report-dir artifacts/visa-hq-report \
  --checkpoint checkpoints/sam_hq_vit_b.pth \
  --workers 1 --device cuda:0
```

Both tools process normal training images. The mask path is `<category>/Data/Foreground/Normal/<image-stem>.png`; preserve the source stem when converting `.JPG` image names to `.png` mask names. Select and refine candidates by source-image comparison, then place the accepted masks under `visa-foreground/` with the same relative paths. The [VisA foreground notes](visa_foreground_baseline.md) record the existing baseline selection and the subsequent 21-image refinement.

## Review and validation

Inspect object edges, separate objects, real holes, transparent material, and cast shadows. Dark object material belongs to foreground; projected shadows and visible background do not. A valid binary file alone does not establish a correct boundary.

Keep each accepted mask's original dimensions and store it as a lossless 0/255 PNG. Generation records and review previews belong outside the final mask directory. `data.foreground_root` and `tools/train.py --foreground-root` select the mask directory; when an external root is set, a missing required mask does not fall back to another location.

HQ-SAM is only needed for mask generation. Training and evaluation do not load it. Evaluation uses the dataset's original anomaly annotations and requires neither training foreground masks nor DTD textures.
