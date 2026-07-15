#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/mvtec.yaml}
DATA_ROOT=${DATA_ROOT:-data/MVTec-AD}
TEXTURE_ROOT=${TEXTURE_ROOT:-data/dtd/images}
EPOCHS=${EPOCHS:-50}
GPU_COUNT=${GPU_COUNT:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
SUPPORT_SAMPLES=${SUPPORT_SAMPLES:-500}
IMAGE_LOG_EVERY=${IMAGE_LOG_EVERY:-1000}
RUN_TAG=${RUN_TAG:-mvtec_c3f_50e_$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-runs/${RUN_TAG}_logs}

DEFAULT_CATEGORIES=(
  bottle cable capsule carpet grid hazelnut leather metal_nut
  pill screw tile toothbrush transistor wood zipper
)
if [[ -n "${CATEGORIES_OVERRIDE:-}" ]]; then
  read -r -a CATEGORIES <<< "${CATEGORIES_OVERRIDE}"
else
  CATEGORIES=("${DEFAULT_CATEGORIES[@]}")
fi

mkdir -p "${LOG_DIR}"
printf "category\tgpu\tstatus\tstarted_at\tended_at\n" > "${LOG_DIR}/status.tsv"

run_category() {
  local category=$1
  local gpu=$2
  local started ended status
  started=$(date -Is)
  echo "[${started}] start category=${category} gpu=${gpu}" | tee -a "${LOG_DIR}/launcher.log"
  set +e
  python -u tools/train.py \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --texture-root "${TEXTURE_ROOT}" \
    --category "${category}" \
    --device "cuda:${gpu}" \
    "experiment.name=${RUN_TAG}" \
    "data.batch_size=${BATCH_SIZE}" \
    "data.num_workers=${NUM_WORKERS}" \
    "train.epochs=${EPOCHS}" \
    "train.support_samples=${SUPPORT_SAMPLES}" \
    "train.log_every=20" \
    "train.image_log_every=${IMAGE_LOG_EVERY}" \
    "train.tensorboard=true" \
    > "${LOG_DIR}/${category}.log" 2>&1
  status=$?
  set -e
  ended=$(date -Is)
  printf "%s\t%s\t%s\t%s\t%s\n" "${category}" "${gpu}" "${status}" "${started}" "${ended}" >> "${LOG_DIR}/status.tsv"
  echo "[${ended}] end category=${category} gpu=${gpu} status=${status}" | tee -a "${LOG_DIR}/launcher.log"
  return "${status}"
}

worker() {
  local worker_id=$1
  local gpu=$((worker_id % GPU_COUNT))
  local i=${worker_id}
  local failed=0
  while [ "${i}" -lt "${#CATEGORIES[@]}" ]; do
    run_category "${CATEGORIES[$i]}" "${gpu}" || failed=1
    i=$((i + GPU_COUNT))
  done
  return "${failed}"
}

overall=0
pids=()
for ((i = 0; i < GPU_COUNT && i < ${#CATEGORIES[@]}; i++)); do
  worker "${i}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}" || overall=1
done

exit "${overall}"
