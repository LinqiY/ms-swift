# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch

from swift.utils import upper_bound, lower_bound
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall


@dataclass
class GemmaTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(
        default_factory=lambda: ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_turn>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_turn>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'])


register_template(GemmaTemplateMeta(LLMTemplateType.gemma))


class PaliGemmaTemplate(Template):
    placeholder_tokens = ['<image>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'vllm':
            self.prompt = ['{{QUERY}}']
            return []
        else:
            self.prompt = ['{{QUERY}}\n']
            return ['<image>' * self.processor.image_seq_length + '<bos>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        raw_image = inputs.images
        processor = self.processor
        if encoded['labels'] is not None:
            # print(f"[DEBUG]encoded['labels']:\n {encoded['labels']}")
            # print(f"[DEBUG]len(encoded['labels']):\n {len(encoded['labels'])}")
            # n = upper_bound(0, len(encoded['labels']) -1, lambda idx: encoded['labels'][idx] == -100)
            n = lower_bound(0, len(encoded['labels']), lambda idx: encoded['labels'][idx] != -100)
            n2 = len(encoded['labels']) - n
            encoded['token_type_ids'] = [0] * n + [1] * n2
        else:
            encoded['token_type_ids'] = [0] * len(encoded['input_ids'])
        if raw_image:
            model_inputs = processor(text='<image>' * len(raw_image), images=raw_image, return_tensors='pt')
            # image_seq_length = self.processor.image_seq_length
            # model_inputs = processor(text='<image>' * image_seq_length, images=raw_image, return_tensors='pt')
            # encoded['pixel_values'] = model_inputs['pixel_values'].to(self.model_info.torch_dtype)
            pixel_values = model_inputs['pixel_values'].to(self.model_info.torch_dtype)
            encoded['pixel_values'] = pixel_values
            # print(f"[DEBUG][Pixel Values] Sample pixel_values shape: {pixel_values.shape}")
            # assert pixel_values.ndim != 5, f'pixel_values维度错误: {pixel_values.shape}, 期望不是5维'
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = {}
        # === Handle pixel_values (images) ===
        pixel_values = []
        for idx, b in enumerate(batch):
            p = b.get('pixel_values')
            p_tensor = torch.tensor(p)
            # print(f'[DEBUG][pixel_values] Sample {idx} original shape: {p_tensor.shape}')
            if p is None:
                continue
            p = torch.tensor(p)
            if p.ndim == 3:
                # [c, h, w] → [1, c, h, w]
                p = p.unsqueeze(0)
            pixel_values.append(p)
        for p in pixel_values:
            assert p.ndim == 4    
        if len(pixel_values) > 0:
            max_frames = max([p.shape[0] for p in pixel_values])
            try:
                c, h, w = pixel_values[0].shape[1:]
            except Exception as e:
                print(e)
                print(pixel_values[0].shape)
                exit(0)

            padded_pixel_values = []
            for p in pixel_values:
                pad_len = max_frames - p.shape[0]
                if pad_len > 0:
                    pad_tensor = torch.zeros((pad_len, c, h, w), dtype=p.dtype)
                    p = torch.cat([p, pad_tensor], dim=0)
                padded_pixel_values.append(p)
                # print(f'[DEBUG][pixel_values] Sample {idx} final shape: {p.shape}')

            res['pixel_values'] = torch.stack(padded_pixel_values, dim=0)  # [bsz, max_n_frame, c, h, w]

        # === Handle image_sizes (optional, same length as pixel_values) ===
        image_sizes = [torch.tensor(b['image_sizes']) for b in batch if b.get('image_sizes') is not None]
        if len(image_sizes) > 0:
            max_frames = max([s.shape[0] for s in image_sizes])
            padded_sizes = []
            for s in image_sizes:
                pad_len = max_frames - s.shape[0]
                if pad_len > 0:
                    pad_tensor = torch.zeros((pad_len, *s.shape[1:]), dtype=s.dtype, device=s.device)
                    s = torch.cat([s, pad_tensor], dim=0)
                padded_sizes.append(s)
            res['image_sizes'] = torch.stack(padded_sizes, dim=0)

        # === Handle pixel_values_videos ===
        pixel_values_videos = [torch.tensor(b['pixel_values_videos']) for b in batch if b.get('pixel_values_videos') is not None]
        if len(pixel_values_videos) > 0:
            max_frames = max([v.shape[0] for v in pixel_values_videos])
            c, h, w = pixel_values_videos[0].shape[1:]

            padded_videos = []
            for v in pixel_values_videos:
                pad_len = max_frames - v.shape[0]
                if pad_len > 0:
                    pad_tensor = torch.zeros((pad_len, c, h, w), dtype=v.dtype, device=v.device)
                    v = torch.cat([v, pad_tensor], dim=0)
                padded_videos.append(v)

            res['pixel_values_videos'] = torch.stack(padded_videos, dim=0)  # [bsz, max_frames, c, h, w]

        return res

    # def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     """Handle multimodal data collation for Gemma3n, including audio features"""
    #     res = super()._data_collator_mm_data(batch)

    #     # Handle audio features like other templates do
    #     input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
    #     input_features_mask = [b['input_features_mask'] for b in batch if b.get('input_features_mask') is not None]

    #     if input_features:
    #         res['input_features'] = torch.concat(input_features)
    #     if input_features_mask:
    #         res['input_features_mask'] = torch.concat(input_features_mask)

    #     return res

register_template(
    TemplateMeta(
        MLLMTemplateType.paligemma,
        prefix=[],
        prompt=['{{QUERY}}\n'],
        chat_sep=None,
        suffix=['<eos>'],
        template_cls=PaliGemmaTemplate,
    ))


@dataclass
class Gemma3TextTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(
        default_factory=lambda: ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_turn>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_turn>'])


class Gemma3Template(Template):

    def _swift_encode(self, inputs: StdTemplateInputs):
        if inputs.system is not None:
            system = inputs.system
            inputs.system = None
            inputs.messages[0]['content'] = system + '\n\n' + inputs.messages[0]['content']
        for message in inputs.messages:
            if message['role'] == 'assistant' and isinstance(message['content'], str):
                message['content'] = message['content'].strip('\n')
        return super()._swift_encode(inputs)


register_template(Gemma3TextTemplateMeta(LLMTemplateType.gemma3_text, template_cls=Gemma3Template))


class Gemma3VisionTemplate(Gemma3Template):
    boi_token_id = 255999
    placeholder_tokens = ['<start_of_image>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<start_of_image>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

        encoded = super()._encode(inputs)
        if inputs.images:
            input_ids = encoded['input_ids']
            labels = encoded['labels']
            idx_list = findall(input_ids, self.boi_token_id)
            img_tokens = self._tokenize(self.processor.full_image_sequence)
            input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, lambda _: img_tokens)

            # TODO: customize
            processor_kwargs = Gemma3ProcessorKwargs._defaults['images_kwargs']
            image_inputs = self.processor.image_processor(inputs.images, **processor_kwargs)
            image_inputs['pixel_values'] = torch.as_tensor(np.array(image_inputs['pixel_values']))
            image_inputs.pop('num_crops')

            array_ids = np.array(input_ids)
            mm_token_type_ids = np.zeros_like(input_ids)
            mm_token_type_ids[array_ids == self.processor.image_token_id] = 1
            encoded['token_type_ids'] = mm_token_type_ids.tolist()
            encoded['input_ids'] = input_ids
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['labels'] = labels
        return encoded


register_template(GemmaTemplateMeta(MLLMTemplateType.gemma3_vision, template_cls=Gemma3VisionTemplate))
