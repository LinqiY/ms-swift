# !/bin/bash

datasets=(
    # AI-ModelScope/LLaVA-Instruct-150K 
    # AI-ModelScope/LaTeX_OCR
    # AI-ModelScope/ShareGPT-4o
    # AI-ModelScope/ShareGPT4V
    # AI-ModelScope/coco
    # modelscope/coco_2014_caption
    # swift/A-OKVQA
    # swift/ChartQA
    # swift/GRIT
    # swift/Multimodal-Mind2Web
    # swift/OCR-VQA
    # swift/RLAIF-V-Dataset
    # swift/ScienceQA
    # swift/gpt4v-dataset
    # swift/llava-instruct-mix-vsft
    # swift/llava-med-zh-instruct-60k
    # swift/lnqa
    # swift/path-vqa
    # swift/refcoco
    # swift/refcocog
    # tany0699/garbage265
    AI-ModelScope/LLaVA-Pretrain
    )


local_dir_base="/inspire/hdd/global_user/gongjingjing-25039/sdzhang/dataset/vl_dataset"
cache_dir="/inspire/hdd/global_user/gongjingjing-25039/sdzhang/ms_cache/"

for dataset in "${datasets[@]}"; do
    echo "Downloading: $dataset"
    modelscope download \
        --dataset "$dataset" \
        --local_dir "$local_dir_base/$(basename "$dataset")" \
        --cache_dir "$cache_dir"
done