#!/bin/bash
set -e 

EXP_NAME="PaliGemma_stage4_affordance"
PROJECT_ROOT="/inspire/hdd/global_user/gongjingjing-25039/lqyin/ms-swift"
cd "${PROJECT_ROOT}"
OUTPUT_DIR="${PROJECT_ROOT}/output/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

MASTER_PORT=29501 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift sft \
    --model  /inspire/hdd/global_user/gongjingjing-25039/lqyin/ms-swift/output/PaliGemma_stage3/v8-20250718-085547/checkpoint-400 \
    --check_model false \
    --train_type lora \
    --dataset ShareRobot/affordance \
    --torch_dtype bfloat16 \
    --max_steps 2000 \
    --streaming true \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate "1e-4" \
    --lora_rank "8" \
    --lora_alpha "32" \
    --gradient_accumulation_steps 1 \
    --packing true \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --save_only_model true \
    --output_dir "${OUTPUT_DIR}" \
    --attn_impl flash_attn \
    --use_liger_kernel false \
    --save_safetensors false \
    --do_eval true \
    --split_dataset_ratio 0.05 \
    --deepspeed zero3 \
    > "${OUTPUT_DIR}/outputs.log" 2>&1
