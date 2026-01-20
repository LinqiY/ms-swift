NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model  /inspire/hdd/global_user/gongjingjing-25039/sdzhang/model/paligemma-3b-pt-224 \
    --check_model false \
    --train_type full \
    --dataset VLABench/affordance VLABench/goal_description VLABench/spatial_understanding VLABench/task_planning VLABench/trajectory \
    --torch_dtype bfloat16 \
    --max_steps 2000 \
    --streaming true \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
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
    --output_dir output/PaliGemma_VLABench \
    --attn_impl flash_attn \
    --use_liger_kernel false \
    --save_safetensors false \
    --do_eval true \
    --split_dataset_ratio 0.05 \
    --deepspeed zero3 \
    > output/PaliGemma_VLABench/outputs.log 2>&1
