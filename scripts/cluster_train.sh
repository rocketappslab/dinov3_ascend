#!/usr/bin/env bash
# DINOv3 Ascend NPU pretraining on the internal cluster.
# Stages weights and images into /cache, then launches scripts/train_npu.sh
# on every worker.
#
# Overrides (job environment or the shell):
#   S3_WEIGHTS=s3://bucket-6417/rocket/model/timm/vit_7b_patch16_dinov3.lvd1689m
#   S3_IMAGE_DIR=s3://bucket-6417/rocket/Dataset/demo.csv
#   S3_IMAGE_LIST=s3://.../a.csv,s3://.../b.json
#   S3_IMAGE_ROOT=s3://bucket-6417/rocket/Dataset/demo   # empty: use folders in the lists
#   IMAGE_DIR=/cache/demo.csv
#   IMAGE_ROOT=/cache/demo
#   IMAGE_LIST=/cache/a.csv,/cache/b.json
#   PRETRAINED_WEIGHTS=/cache/vit_7b_patch16_dinov3.lvd1689m
#   OUTPUT_DIR=/cache/exp/dinov3_vit7b16_pretrain_npu
#
# Checkpoints are written with distributed checkpointing, so OUTPUT_DIR has to
# be visible to every node.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

S3_BUCKET="${S3_BUCKET:-s3://bucket-6417/rocket}"
S3_WEIGHTS="${S3_WEIGHTS:-${S3_BUCKET}/model/timm/vit_7b_patch16_dinov3.lvd1689m}"
S3_IMAGE_DIR="${S3_IMAGE_DIR:-${S3_BUCKET}/Dataset/demo.csv}"
S3_IMAGE_LIST="${S3_IMAGE_LIST:-}"
S3_IMAGE_ROOT="${S3_IMAGE_ROOT-${S3_BUCKET}/Dataset/demo}"

CACHE_ROOT="${CACHE_ROOT:-/cache}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-${CACHE_ROOT}/vit_7b_patch16_dinov3.lvd1689m}"
IMAGE_DIR="${IMAGE_DIR:-}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
IMAGE_LIST="${IMAGE_LIST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${CACHE_ROOT}/exp/dinov3_vit7b16_pretrain_npu}"

copy_obs() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "${dst}")"
    echo "[MOX] ${src} -> ${dst}"
    if [[ -f "${ROOT}/scripts/train_ascend/mox_cp.py" ]]; then
        python "${ROOT}/scripts/train_ascend/mox_cp.py" "${src}" "${dst}"
    else
        python -c 'import moxing as mox, sys; mox.file.copy_parallel(sys.argv[1], sys.argv[2])' "${src}" "${dst}"
    fi
}

if [[ ! -e "${PRETRAINED_WEIGHTS}" ]]; then
    copy_obs "${S3_WEIGHTS}" "${PRETRAINED_WEIGHTS}"
fi

if [[ -z "${IMAGE_DIR}" ]]; then
    IMAGE_DIR="${CACHE_ROOT}/$(basename "${S3_IMAGE_DIR%/}")"
fi
if [[ ! -e "${IMAGE_DIR}" ]]; then
    copy_obs "${S3_IMAGE_DIR}" "${IMAGE_DIR}"
fi

if [[ -z "${IMAGE_ROOT}" && -n "${S3_IMAGE_ROOT}" ]]; then
    IMAGE_ROOT="${CACHE_ROOT}/$(basename "${S3_IMAGE_ROOT%/}")"
fi
if [[ -n "${IMAGE_ROOT}" && ! -e "${IMAGE_ROOT}" ]]; then
    copy_obs "${S3_IMAGE_ROOT}" "${IMAGE_ROOT}"
fi

if [[ -z "${IMAGE_LIST}" && -n "${S3_IMAGE_LIST}" ]]; then
    IFS=',' read -r -a S3_LISTS <<< "${S3_IMAGE_LIST}"
    for src in "${S3_LISTS[@]}"; do
        src="${src#"${src%%[![:space:]]*}"}"
        src="${src%"${src##*[![:space:]]}"}"
        if [[ -z "${src}" ]]; then
            continue
        fi
        dst="${CACHE_ROOT}/$(basename "${src%/}")"
        if [[ ! -e "${dst}" ]]; then
            copy_obs "${src}" "${dst}"
        fi
        IMAGE_LIST="${IMAGE_LIST:+${IMAGE_LIST},}${dst}"
    done
fi

# Cluster worker environment. Do not rename these variables.
MASTER_HOST="${VC_WORKER_HOSTS:?VC_WORKER_HOSTS is required}"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="${MA_NUM_HOSTS:?MA_NUM_HOSTS is required}"
NODE_RANK="${VC_TASK_INDEX:?VC_TASK_INDEX is required}"
NGPUS_PER_NODE="${MA_NUM_GPUS:?MA_NUM_GPUS is required}"
NUM_PROCESSES=$((NGPUS_PER_NODE * NNODES))
MASTER_PORT="${MASTER_PORT:-6060}"
JOB_ID="${JOB_ID:-${MA_VJ_NAME:-1234}}"

echo "------> system config <------"
echo "VC_WORKER_HOSTS: ${VC_WORKER_HOSTS}"
echo "MASTER_HOST: ${MASTER_HOST}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NGPUS_PER_NODE: ${NGPUS_PER_NODE}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "MA_JOB_DIR: ${MA_JOB_DIR:-}"
echo "output S3 url: ${train_url:-}"
echo "------>  <------"

export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1800

if [[ -n "${MA_VJ_NAME:-}" && -n "${MA_LOG_DIR:-}" ]]; then
    ma_vj_name="${MA_VJ_NAME//ma-job/modelarts-job}"
    task_name="worker-${VC_TASK_INDEX}"
    task_plog_path="${MA_LOG_DIR}/${ma_vj_name}/${task_name}"
    mkdir -p "${task_plog_path}"
    export ASCEND_PROCESS_LOG_PATH="${task_plog_path}"
    echo "plog path: ${ASCEND_PROCESS_LOG_PATH}"
fi

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    devices=""
    for ((i = 0; i < NGPUS_PER_NODE; i++)); do
        devices="${devices:+${devices},}${i}"
    done
    export ASCEND_RT_VISIBLE_DEVICES="${devices}"
fi

export IMAGE_DIR IMAGE_ROOT IMAGE_LIST OUTPUT_DIR PRETRAINED_WEIGHTS
export NPROC_PER_NODE="${NGPUS_PER_NODE}"
export MASTER_PORT NNODES NODE_RANK MASTER_ADDR
export RDZV_ID="${JOB_ID}"
export RDZV_CONF="${RDZV_CONF:-timeout=1800}"

echo "------> pwd <------"
pwd
echo "------> files <------"
ls

exec bash "${ROOT}/scripts/train_npu.sh"
