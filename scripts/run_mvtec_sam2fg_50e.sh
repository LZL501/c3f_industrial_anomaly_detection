#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data/MVTec-AD}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FOREGROUND_GPU=${FOREGROUND_GPU:-0}
FOREGROUND_BACKEND=${FOREGROUND_BACKEND:-sam_hq}
FOREGROUND_STRATEGY=${FOREGROUND_STRATEGY:-v2}
SAM2_CHECKPOINT=${SAM2_CHECKPOINT:-checkpoints/sam2.1_hiera_base_plus.pt}
SAM2_CONFIG=${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_b+.yaml}
SAM_HQ_CHECKPOINT=${SAM_HQ_CHECKPOINT:-checkpoints/sam_hq_vit_b.pth}
SAM_MODEL_TYPE=${SAM_MODEL_TYPE:-vit_b}
FOREGROUND_HQ_TOKEN_ONLY=${FOREGROUND_HQ_TOKEN_ONLY:-1}
PREVIEW_COUNT=${PREVIEW_COUNT:-4}

if [[ -z "${FOREGROUND_CHECKPOINT:-}" ]]; then
  if [[ "$FOREGROUND_BACKEND" == "sam_hq" ]]; then
    FOREGROUND_CHECKPOINT="$SAM_HQ_CHECKPOINT"
  else
    FOREGROUND_CHECKPOINT="$SAM2_CHECKPOINT"
  fi
fi

PYTHONPATH_PARTS=()
if [[ -d third_party/python_deps ]]; then
  PYTHONPATH_PARTS+=(third_party/python_deps)
fi
if [[ "$FOREGROUND_BACKEND" == "sam_hq" ]]; then
  PYTHONPATH_PARTS+=(third_party/sam_hq)
elif [[ "$FOREGROUND_BACKEND" == "sam2" ]]; then
  PYTHONPATH_PARTS+=(third_party/sam2)
fi
FOREGROUND_PYTHONPATH=$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")
if [[ -n "${PYTHONPATH:-}" ]]; then
  FOREGROUND_PYTHONPATH="${FOREGROUND_PYTHONPATH:+${FOREGROUND_PYTHONPATH}:}${PYTHONPATH}"
fi

HQ_TOKEN_ARGS=()
if [[ "$FOREGROUND_HQ_TOKEN_ONLY" == "0" ]]; then
  HQ_TOKEN_ARGS+=(--no-hq-token-only)
fi

CATEGORIES=(
  bottle cable capsule carpet grid hazelnut leather metal_nut pill screw
  tile toothbrush transistor wood zipper
)

PIPELINE_LOG="runs/${RUN_TAG}_pipeline.log"
FOREGROUND_LOG="runs/${RUN_TAG}_foreground.log"
COUNTS_LOG="runs/${RUN_TAG}_foreground_counts.tsv"

mkdir -p runs
echo "foreground_start $(date --iso-8601=seconds)" | tee "$PIPELINE_LOG"

PYTHONPATH="$FOREGROUND_PYTHONPATH" CUDA_VISIBLE_DEVICES="$FOREGROUND_GPU" python tools/generate_mvtec_foreground_sam.py \
  --backend "$FOREGROUND_BACKEND" \
  --strategy "$FOREGROUND_STRATEGY" \
  --prompt-mode heuristic_box \
  --root "$DATA_ROOT" \
  --checkpoint "$FOREGROUND_CHECKPOINT" \
  --sam2-config "$SAM2_CONFIG" \
  --model-type "$SAM_MODEL_TYPE" \
  --categories all \
  --preview-dir "runs/${RUN_TAG}_foreground_preview" \
  --preview-count "$PREVIEW_COUNT" \
  "${HQ_TOKEN_ARGS[@]}" \
  > "$FOREGROUND_LOG" 2>&1

{
  echo -e "category\tforeground\tgood"
  for category in "${CATEGORIES[@]}"; do
    good_count=$(find "$DATA_ROOT/$category/train/good" -maxdepth 1 -type f -name "*.png" | wc -l)
    foreground_count=$(find "$DATA_ROOT/$category/train/foreground" -maxdepth 1 -type f -name "*.png" | wc -l)
    echo -e "${category}\t${foreground_count}\t${good_count}"
    if [[ "$foreground_count" -ne "$good_count" ]]; then
      echo "foreground count mismatch for ${category}: ${foreground_count}/${good_count}" >&2
      exit 1
    fi
  done
} | tee "$COUNTS_LOG"

echo "train_start $(date --iso-8601=seconds)" | tee -a "$PIPELINE_LOG"
set +e
bash scripts/run_mvtec_all_50e.sh
status=$?
set -e
echo "pipeline_end $(date --iso-8601=seconds) status=${status}" | tee -a "$PIPELINE_LOG"
exit "$status"
