

python scripts/train_ascend/mox_cp.py s3://bucket-6417/rocket/Huggingface/LLaDA-8B-Instruct-HF/ /cache/LLaDA-8B-Instruct-HF/
python scripts/train_ascend/mox_cp.py s3://bucket-6417/rocket/Huggingface/siglip2-so400m-patch14-384/ /cache/siglip2-so400m-patch14-384/

LLM_VERSION="/cache/LLaDA-8B-Instruct-HF/"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/cache/siglip2-so400m-patch14-384/"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
echo "output S3 url: ${train_url}"

############### v1 ################
python scripts/train_ascend/mox_cp.py s3://bucket-6417/rocket/Dataset/dVLM/LLaVA-Pretrain/images_subset/ /cache/images_subset/
python scripts/train_ascend/mox_cp.py s3://bucket-6417/rocket/Dataset/dVLM/LLaVA-Pretrain/json_subset/blip_laion_cc_sbu_558k_subset.json /cache/blip_laion_cc_sbu_558k_subset.json
IMG_PATH=/cache/images_subset/
DATA_PATH=/cache/blip_laion_cc_sbu_558k_subset.json
############### v1 ################




# 系统默认环境变量，不建议修改
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
NGPUS_PER_NODE="$MA_NUM_GPUS"
NUM_PROCESSES=$(($NGPUS_PER_NODE * $NNODES))

MASTER_PORT="6060"
JOB_ID="1234"

echo "------> system config <------"
echo "VC_WORKER_HOSTS: ${VC_WORKER_HOSTS}"
echo "MASTER_HOST: ${MASTER_HOST}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NGPUS_PER_NODE: ${NGPUS_PER_NODE}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "${MA_JOB_DIR}"
echo "------>  <------"

export HCCL_WHITELIST_DISABLE=1

if [[ $NODE_RANK == 0 ]]; then
    EXT_ARGS="--rdzv_conf=is_host=1"
else
    EXT_ARGS=""
fi

ma_vj_name=`echo ${MA_VJ_NAME} | sed 's:ma-job:modelarts-job:g'`
task_name="worker-${VC_TASK_INDEX}"
task_plog_path=${MA_LOG_DIR}/${ma_vj_name}/${task_name}

mkdir -p ${task_plog_path}
export ASCEND_PROCESS_LOG_PATH=${task_plog_path}

echo "plog path: ${ASCEND_PROCESS_LOG_PATH}"

# set hccl timeout time in seconds
export HCCL_CONNECT_TIMEOUT=1800

echo "------> pwd <------"
pwd
echo "------> files <------"
ls

############### Pretrain ################

PROMPT_VERSION=llada_plain

BASE_RUN_NAME="llada_v_pretrain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

export TOKENIZERS_PARALLELISM=false
# python -m debugpy --listen 5678 --wait-for-client -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}"  --master_port="${MASTER_PORT}" \
# python -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}"  --master_port="${MASTER_PORT}" \
#accelerate launch \
# torchrun --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     $EXT_ARGS \
#     --nproc_per_node=$NGPUS_PER_NODE \
#     --rdzv_id=$JOB_ID \
#     --rdzv_backend=static \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
accelerate launch \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMG_PATH} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /cache/exp/${BASE_RUN_NAME} \
    --num_train_epochs 5000 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa
