import torch.distributed as dist
import os
from torch.utils.data import DataLoader
from typing import List
from functools import partial
from swift.llm.dataset import load_dataset
from swift.llm.argument import BaseArguments, TrainArguments
from swift.llm.dataset import EncodePreprocessor
from swift.llm.data_loader import BatchSamplerShard, DataLoaderShard, DataLoaderDispatcher
# Optional dataset:
# ms
# AI-ModelScope/LLaVA-Instruct-150K
# AI-ModelScope/LaTeX_OCR
# AI-ModelScope/M3IT # HUGE
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
# hf
# lmms-lab/GQA
def load_vl_dataloader(args: TrainArguments, 
                       train_dataset,
                       val_dataset,
                       processor=None,
                       debug: bool = False):
    if processor is None:
        _, processor = args.get_model_processor()
    
    assert processor is not None
    # load processor as template
    template = args.get_template(processor)
    template.set_mode("train")
    
    if debug:
        train_dataset = train_dataset.select(range(1000))
        val_dataset = val_dataset.select(range(100))
    processor = EncodePreprocessor(template)
    train_dataset = processor(train_dataset, num_proc=args.dataset_num_proc, strict=args.strict)
    val_dataset = processor(val_dataset, num_proc=args.dataset_num_proc, strict=args.strict)
    # load collate function and dataloader
    padding_to = args.max_length if args.train_type == 'longlora' else None
    data_collator = partial(template.data_collator, padding_to=padding_to)
    dataloader_params = {
        'collate_fn': data_collator,
        'num_workers': args.dataloader_num_workers,
        'pin_memory': args.dataloader_pin_memory,
        'persistent_workers': args.dataloader_persistent_workers,
        'prefetch_factor': args.dataloader_prefetch_factor
    }
    batch_sampler_params = {
        'drop_last': args.dataloader_drop_last,
        'shuffle': args.train_dataloader_shuffle,
        'data_seed': args.data_seed,
    }

    if hasattr(train_dataset, '__len__') and hasattr(val_dataset, '__len__'):
        train_batch_sampler = BatchSamplerShard(
            len(train_dataset), batch_size=args.per_device_train_batch_size, **batch_sampler_params)
        val_batch_sampler = BatchSamplerShard(
            len(val_dataset), batch_size=args.per_device_train_batch_size, **batch_sampler_params)
        train_dataloader = DataLoaderShard(train_dataset, train_batch_sampler, **dataloader_params)
        val_dataloader = DataLoaderShard(val_dataset, val_batch_sampler, **dataloader_params)
    
    else: # Iterable dataset
        if dist.is_initialized():
            dataloader_params['prefetch_factor'] = dataloader_params['prefetch_factor'] * dist.get_world_size()
        train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, **dataloader_params)
        train_dataloader = DataLoaderDispatcher(train_dataloader)
        val_dataloader = DataLoader(val_dataset, batch_size=args.per_device_train_batch_size, **dataloader_params)
        val_dataloader = DataLoaderDispatcher(val_dataloader)
        
    return train_dataloader, val_dataloader

def get_overide_arguments(
    model_id_or_path: str,
    **kwargs
):
    assert model_id_or_path, "Model ID or path must be provided."
    args = TrainArguments(
        model=model_id_or_path,
        **kwargs
    )
    return args

if __name__ == "__main__":
    ms_cache = 'datasets'
    datasets = [
        # ms
        # 'AI-ModelScope/LLaVA-Instruct-150K',
        # 'AI-ModelScope/LaTeX_OCR',
        # 'AI-ModelScope/ShareGPT-4o',
        # "AI-ModelScope/ShareGPT4V:ShareGPT4V/ShareGPT4V-PT"
        # 'swift/A-OKVQA',
        # 'swift/ChartQA',
        # 'swift/OCR-VQA',
        # 'swift/RLAIF-V-Dataset',
        # 'swift/ScienceQA',
        # 'swift/path-vqa'
        ## hf
        # 'linxy/LaTeX_OCR',
        # 'detection-datasets/coco'
        # 'OpenGVLab/ShareGPT-4o'
        # 'ShareRobot/planning_test'
        # 'VLAOSDataset/planning'
        # 'VLAOSDataset/subtask'
        # 'VLAOSDataset/move'
        # 'VLAOSDataset/bbox'
        # 'VLAOSDataset/gripper_position'
        # 'VLABench/affordance',
        # 'VLABench/goal_description',
        # 'VLABench/spatial_understanding',
        # 'VLABench/task_planning_test',
        # 'VLABench/trajectory',
        # 'FruitPickAndPlaceVQA',
        "WidowX-VQA/ood-gemini",
        # "WidowX-VQA/ood-trajectory",
        # "WidowX-VQA/trajectory",
        # "WidowX-VQA/gemini",
    ]
    train_dataset, val_dataset = load_dataset(
        datasets,
        split_dataset_ratio=0.05,
        use_hf=False,
        # hub_token='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'
    )
    train_dataset = train_dataset.select(range(5))  # 选择前5条数据
    val_dataset = val_dataset.select(range(5))  # 选择前5条数据
    
    dataloader_args = get_overide_arguments(
        model_id_or_path="/inspire/hdd/global_user/gongjingjing-25039/sdzhang/model/paligemma-3b-pt-224/",
        # model_id_or_path="/inspire/hdd/global_user/gongjingjing-25039/sdzhang/model/Qwen2.5-VL-3B-Instruct",
        dataset=datasets,
        per_device_train_batch_size=4,
        dataloader_num_workers=0,
    ) 
    train_vl_dataloader, val_vl_dataloader = load_vl_dataloader(
        dataloader_args, train_dataset, val_dataset, processor=None
    )
    import pdb; pdb.set_trace()