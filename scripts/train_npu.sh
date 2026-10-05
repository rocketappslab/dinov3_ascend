#!/usr/bin/env bash
# Train DINOv3 on Ascend NPUs.
#
#   bash scripts/train_npu.sh
#
# Overrides:
#   IMAGE_DIR=/path/to/images NPROC_PER_NODE=8 bash scripts/train_npu.sh
#   IMAGE_DIR=/path/to/images IMAGE_LIST=/path/a.csv,/path/b.json bash scripts/train_npu.sh
#   IMAGE_DIR=/path/to/paths.json bash scripts/train_npu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
set +u
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

IMAGE_DIR="${IMAGE_DIR:-/root/workspace/rocket/Dataset/demo.csv}"
IMAGE_LIST="${IMAGE_LIST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/dinov3_vit7b16_pretrain_npu}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29511}"

if [[ -n "${IMAGE_LIST}" ]]; then
    IFS=',' read -r -a IMAGE_LISTS <<< "${IMAGE_LIST}"
    for list in "${IMAGE_LISTS[@]}"; do
        list="${list#"${list%%[![:space:]]*}"}"
        list="${list%"${list##*[![:space:]]}"}"
        if [[ -z "${list}" ]]; then
            continue
        fi
        if [[ -d "${list}" ]]; then
            continue
        fi
        if [[ ! -f "${list}" ]]; then
            echo "Image list does not exist: ${list}" >&2
            exit 1
        fi
        case "${list}" in
            *.csv|*.json|*.CSV|*.JSON) ;;
            *)
                echo "Image list must be .csv or .json: ${list}" >&2
                exit 1
                ;;
        esac
    done
    if [[ ! -d "${IMAGE_DIR}" ]]; then
        echo "Image directory does not exist: ${IMAGE_DIR}" >&2
        exit 1
    fi
    DATASET_PATH="ImageDir:root=${IMAGE_DIR}:extra=${IMAGE_LIST}"
elif [[ -d "${IMAGE_DIR}" || -f "${IMAGE_DIR}" || "${IMAGE_DIR}" == *,* ]]; then
    DATASET_PATH="ImageDir:root=${IMAGE_DIR}"
else
    echo "Image directory or list does not exist: ${IMAGE_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# iBOT always calls module.compile(); NPU inductor needs Triton, which is not installed.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

echo "[PY] $(command -v python)"
echo "[IMAGES] ${DATASET_PATH}"
echo "[OUT] ${OUTPUT_DIR}"
echo "[NPROC] ${NPROC_PER_NODE}"
echo "[DEVICES] ${ASCEND_RT_VISIBLE_DEVICES}"

exec python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    scripts/train_npu.py \
    --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_npu.yaml \
    --output-dir "${OUTPUT_DIR}" \
    "train.dataset_path=${DATASET_PATH}"
