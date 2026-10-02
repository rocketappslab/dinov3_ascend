#!/usr/bin/env bash
# Train DINOv3 on Ascend NPUs.
#
#   bash scripts/train_npu.sh
#
# Overrides:
#   IMAGE_DIR=/path/to/images NPROC_PER_NODE=8 bash scripts/train_npu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
set +u
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

IMAGE_DIR="${IMAGE_DIR:-/root/workspace/rocket/Dataset/demo}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/demo_npu}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"

if [[ ! -d "${IMAGE_DIR}" ]]; then
    echo "Image directory does not exist: ${IMAGE_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
# iBOT always calls module.compile(); NPU inductor needs Triton, which is not installed.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "[PY] $(command -v python)"
echo "[IMAGES] ${IMAGE_DIR}"
echo "[OUT] ${OUTPUT_DIR}"
echo "[NPROC] ${NPROC_PER_NODE}"
echo "[DEVICES] ${ASCEND_RT_VISIBLE_DEVICES}"

exec python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    scripts/train_npu.py \
    --config-file dinov3/configs/train/demo_npu.yaml \
    --output-dir "${OUTPUT_DIR}" \
    "train.dataset_path=ImageDir:root=${IMAGE_DIR}"
