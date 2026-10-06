#!/usr/bin/env bash
# Train DINOv3 on Ascend NPUs.
#
#   bash scripts/train_npu.sh
#
# Overrides:
#   IMAGE_DIR=/path/to/images NPROC_PER_NODE=8 bash scripts/train_npu.sh
#   IMAGE_DIR=/path/to/images IMAGE_LIST=/path/a.csv,/path/b.json bash scripts/train_npu.sh
#   IMAGE_DIR=/path/to/paths.json bash scripts/train_npu.sh
#   IMAGE_DIR=/path/to/list.csv   # rows pair an image folder root with a relative path
#   IMAGE_DIR=/path/to/list.csv IMAGE_ROOT=/path/to/images bash scripts/train_npu.sh
#     # IMAGE_ROOT is joined with each relative path in the IMAGE_DIR / IMAGE_LIST lists
#   PRETRAINED_WEIGHTS=/path/to/timm_dinov3 bash scripts/train_npu.sh
#
# Multi-node (scripts/cluster_train.sh sets these from the cluster job):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<host> MASTER_PORT=6060 NPROC_PER_NODE=8 bash scripts/train_npu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv/bin/activate"
fi
set +u
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

IMAGE_DIR="${IMAGE_DIR:-/root/workspace/rocket/Dataset/demo2.csv}"
# Set IMAGE_ROOT= (empty) to use the folders stored in the lists instead.
IMAGE_ROOT="${IMAGE_ROOT-/root/workspace/rocket/Dataset/demo}"
IMAGE_LIST="${IMAGE_LIST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/dinov3_vit7b16_pretrain_npu}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-/root/workspace/rocket/Model/timm/vit_7b_patch16_dinov3.lvd1689m/}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
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

if [[ -n "${IMAGE_ROOT}" ]]; then
    if [[ ! -d "${IMAGE_ROOT}" ]]; then
        echo "Image folder root does not exist: ${IMAGE_ROOT}" >&2
        exit 1
    fi
    DATASET_PATH="${DATASET_PATH}:image_root=${IMAGE_ROOT}"
fi

if [[ -n "${PRETRAINED_WEIGHTS}" && ! -e "${PRETRAINED_WEIGHTS}" ]]; then
    echo "Pretrained weights do not exist: ${PRETRAINED_WEIGHTS}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# iBOT always calls module.compile(); NPU inductor needs Triton, which is not installed.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

TRAIN_OPTS=("train.dataset_path=${DATASET_PATH}")
if [[ -n "${PRETRAINED_WEIGHTS}" ]]; then
    TRAIN_OPTS+=("student.pretrained_weights=${PRETRAINED_WEIGHTS}")
fi

DIST_ARGS=(
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
)
if [[ -n "${RDZV_ID:-}" ]]; then
    DIST_ARGS+=(--rdzv_id="${RDZV_ID}")
fi
if [[ -n "${RDZV_CONF:-}" ]]; then
    DIST_ARGS+=(--rdzv_conf="${RDZV_CONF}")
fi

echo "[PY] $(command -v python)"
echo "[IMAGES] ${DATASET_PATH}"
echo "[IMAGE_ROOT] ${IMAGE_ROOT:-<from lists>}"
echo "[WEIGHTS] ${PRETRAINED_WEIGHTS:-<config>}"
echo "[OUT] ${OUTPUT_DIR}"
echo "[NPROC] ${NPROC_PER_NODE}"
echo "[NNODES] ${NNODES}"
echo "[NODE_RANK] ${NODE_RANK}"
echo "[MASTER] ${MASTER_ADDR}:${MASTER_PORT}"
echo "[DEVICES] ${ASCEND_RT_VISIBLE_DEVICES}"

exec python -m torch.distributed.run \
    "${DIST_ARGS[@]}" \
    scripts/train_npu.py \
    --config-file dinov3/configs/train/dinov3_vit7b16_pretrain_npu.yaml \
    --output-dir "${OUTPUT_DIR}" \
    "${TRAIN_OPTS[@]}"
