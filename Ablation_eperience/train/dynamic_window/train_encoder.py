import torch
import os
import sys
import gc
import json
import logging
import random
import math
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    PreTrainedModel, 
    Trainer, 
    TrainingArguments,
    PreTrainedTokenizerBase,
    TrainerCallback
)
from transformers.tokenization_utils_base import PaddingStrategy
from scipy import spatial
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用相对路径导入data_converter（如果不需要可以注释）
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, 'utils')
sys.path.insert(0, utils_path)
try:
    from data_converter import TripletDataConverter
except ImportError:
    TripletDataConverter = None
    logger.warning("未找到 data_converter，原有数据加载方法将不可用")

# ============================================================================
# 1. 工具函数
# ============================================================================

def set_device() -> str:
    """自动设置设备"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def find_all_linear_names(model: PreTrainedModel, linear_type: Optional[object] = None) -> List[str]:
    """查找所有线性层名称"""
    if linear_type is None:
        linear_type = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_type):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_pooling(outputs: torch.Tensor,
                inputs: Dict,
                pooling_strategy: str,
                padding_side: str) -> torch.Tensor:
    """池化模型输出"""
    if pooling_strategy == 'cls':
        outputs = outputs[:, 0]
    elif pooling_strategy == 'cls_avg':
        avg = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / inputs["attention_mask"].sum(dim=1).unsqueeze(1)
        outputs = (outputs[:, 0] + avg) / 2.0
    elif pooling_strategy == 'cls_max':
        maximum, _ = torch.max(outputs * inputs["attention_mask"][:, :, None], dim=1)
        outputs = (outputs[:, 0] + maximum) / 2.0
    elif pooling_strategy == 'last':
        batch_size = inputs['input_ids'].shape[0]
        sequence_lengths = -1 if padding_side == 'left' else inputs["attention_mask"].sum(dim=1) - 1
        outputs = outputs[torch.arange(batch_size, device=outputs.device), sequence_lengths]
    elif pooling_strategy in ['avg', 'mean']:
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / inputs["attention_mask"].sum(dim=1).unsqueeze(1)
    elif pooling_strategy == 'max':
        outputs, _ = torch.max(outputs * inputs["attention_mask"][:, :, None], dim=1)
    elif pooling_strategy == 'all':
        # keep outputs
        pass
    elif isinstance(pooling_strategy, int) or (isinstance(pooling_strategy, str) and pooling_strategy.isnumeric()):
        # index
        outputs = outputs[:, int(pooling_strategy)]
    else:
        raise NotImplementedError(
            'please specify pooling_strategy from '
            '[`cls`, `cls_avg`, `cls_max`, `last`, `avg`, `mean`, `max`, `all`, int]')
    return outputs

# ============================================================================
# 2. 复数投影层 (AP-Proj)
# ============================================================================
class ComplexProjection(nn.Module):
    """残差可学习复数空间投影层 (Residual AP-Proj)"""
    def __init__(self, hidden_size):
        super().__init__()
        self.half_size = hidden_size // 2
        self.proj_re = nn.Linear(self.half_size, self.half_size)
        self.proj_im = nn.Linear(self.half_size, self.half_size)
        # 零初始化，保证初始时不影响预训练向量
        nn.init.zeros_(self.proj_re.weight)
        nn.init.zeros_(self.proj_im.weight)
        nn.init.zeros_(self.proj_re.bias)
        nn.init.zeros_(self.proj_im.bias)
        
    def forward(self, x):
        re_orig, im_orig = torch.chunk(x, 2, dim=1)
        re = re_orig + self.proj_re(re_orig)
        im = im_orig + self.proj_im(im_orig)
        return torch.cat([re, im], dim=1)

# ============================================================================
# 3. 损失函数
# ============================================================================
def categorical_crossentropy_loss(y_true: torch.Tensor, y_pred: torch.Tensor, from_logits: bool = True) -> torch.Tensor:
    if from_logits:
        return -(F.log_softmax(y_pred, dim=1) * y_true).sum(dim=1)
    return -(torch.log(y_pred, dim=1) * y_true).sum(dim=1)

def cosine_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()
    y_pred = F.normalize(y_pred, p=2, dim=1)
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)

def angle_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 1.0, pooling_strategy: str = 'sum'):
    y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()
    y_pred_re, y_pred_im = torch.chunk(y_pred, 2, dim=1)
    a = y_pred_re[::2]
    b = y_pred_im[::2]
    c = y_pred_re[1::2]
    d = y_pred_im[1::2]
    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z
    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True)**0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True)**0.5
    re /= (dz / dw)
    im /= (dz / dw)
    y_pred = torch.concat((re, im), dim=1)
    if pooling_strategy == 'sum':
        pooling = torch.sum(y_pred, dim=1)
    elif pooling_strategy == 'mean':
        pooling = torch.mean(y_pred, dim=1)
    else:
        raise ValueError(f'Unsupported pooling strategy: {pooling_strategy}')
    y_pred = torch.abs(pooling) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)

def in_batch_negative_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    tau: float = 20.0,
    negative_weights: float = 0.0
) -> torch.Tensor:
    device = y_true.device
    def make_target_matrix(y_true: torch.Tensor):
        idxs = torch.arange(0, y_pred.shape[0]).int().to(device)
        y_true = y_true.int()
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        idxs_1 *= y_true.T
        idxs_1 += (y_true.T == 0).int() * -2
        idxs_2 *= y_true
        idxs_2 += (y_true == 0).int() * -1
        y_true = (idxs_1 == idxs_2).float()
        return y_true
    neg_mask = make_target_matrix(y_true == 0)
    y_true = make_target_matrix(y_true)
    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = y_pred @ y_pred.T
    similarities = similarities - torch.eye(y_pred.shape[0]).to(device) * 1e12
    similarities = similarities * tau
    if negative_weights > 0:
        similarities += neg_mask * negative_weights
    return categorical_crossentropy_loss(y_true, similarities, from_logits=True).mean()

def contrastive_with_negative_loss(
    text: torch.Tensor,
    pos: torch.Tensor,
    neg: Optional[torch.Tensor] = None,
    tau: float = 20.0
) -> torch.Tensor:
    target = torch.cat((pos, neg), dim=0) if neg is not None else pos
    q_norm = F.normalize(text, p=2, dim=1)
    t_norm = F.normalize(target, p=2, dim=1)
    scores = torch.mm(q_norm, t_norm.transpose(0, 1)) * tau
    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
    return nn.CrossEntropyLoss()(scores, labels)

class AngleLoss:
    def __init__(self,
                 cosine_w: float = 0.0,
                 ibn_w: float = 1.0,
                 cln_w: float = 1.0,
                 angle_w: float = 0.02,
                 cosine_tau: float = 20.0,
                 ibn_tau: float = 20.0,
                 angle_tau: float = 20.0,
                 angle_pooling_strategy: str = 'sum',
                 dataset_format: Optional[str] = None,
                 **kwargs):
        self.cosine_w = cosine_w
        self.ibn_w = ibn_w
        self.cln_w = cln_w
        self.angle_w = angle_w
        self.cosine_tau = cosine_tau
        self.ibn_tau = ibn_tau
        self.angle_tau = angle_tau
        self.angle_pooling_strategy = angle_pooling_strategy
        self.dataset_format = dataset_format

    def __call__(self,
                 labels: torch.Tensor,
                 outputs: torch.Tensor) -> torch.Tensor:
        if self.dataset_format == 'A':
            loss = 0.
            if self.cosine_w > 0:
                loss += self.cosine_w * cosine_loss(labels, outputs, self.cosine_tau)
            if self.ibn_w > 0:
                loss += self.ibn_w * in_batch_negative_loss(labels, outputs, self.ibn_tau)
            if self.angle_w > 0:
                loss += self.angle_w * angle_loss(labels, outputs, self.angle_tau,
                                                  pooling_strategy=self.angle_pooling_strategy)
        elif self.dataset_format == 'B':
            query = outputs[::2]
            positive = outputs[1::2]
            loss = contrastive_with_negative_loss(query, positive, neg=None, tau=self.ibn_tau)
        elif self.dataset_format == 'C':
            if int(self.cln_w) == 0:
                logger.info('`cln_w` is set to zero. Contrastive learning with hard negative is disabled.')
            query = outputs[::3]
            positive = outputs[1::3]
            negative = outputs[2::3]
            assert query.shape == positive.shape == negative.shape, f'query.shape={query.shape}, positive.shape={positive.shape}, negative.shape={negative.shape}'
            _, fea_dim = query.shape
            positive_inputs = torch.stack((query, positive), dim=1).reshape(-1, fea_dim)
            positive_labels = torch.ones_like(positive_inputs[:, :1]).long()
            negative_inputs = torch.stack((query, negative), dim=1).reshape(-1, fea_dim)
            negative_labels = torch.zeros_like(negative_inputs[:, :1]).long()
            combined_inputs = torch.cat((positive_inputs, negative_inputs), dim=0)
            combined_labels = torch.cat((positive_labels, negative_labels), dim=0)
            loss = 0.
            cll = 0.
            if self.ibn_w > 0:
                cll += self.ibn_w * contrastive_with_negative_loss(query, positive, tau=self.ibn_tau)
            if self.cln_w > 0:
                cll += self.cln_w * contrastive_with_negative_loss(query, positive, negative, tau=self.ibn_tau)
            if cll > 0:
                loss += cll / 2
            if self.angle_w > 0:
                loss += self.angle_w * angle_loss(combined_labels, combined_inputs, self.angle_tau,
                                                  pooling_strategy=self.angle_pooling_strategy)
            if self.cosine_w > 0:
                loss += self.cosine_w * cosine_loss(combined_labels, combined_inputs, self.cosine_tau)
        else:
            raise NotImplementedError
        return loss

# ============================================================================
# 4. Pooler类
# ============================================================================
class Pooler:
    def __init__(self,
                 model: PreTrainedModel,
                 pooling_strategy: Optional[str] = None,
                 padding_side: Optional[str] = None):
        self.model = model
        self.pooling_strategy = pooling_strategy
        self.padding_side = padding_side

    def __call__(self,
                 inputs: Dict,
                 layer_index: int = -1,
                 embedding_start: Optional[int] = None,
                 embedding_size: Optional[int] = None,
                 return_all_layer_outputs: bool = False,
                 pooling_strategy: Optional[str] = None,
                 return_mlm_logits: bool = False) -> torch.Tensor:
        if layer_index == -1 and not return_all_layer_outputs:
            ret = self.model(output_hidden_states=True, **inputs)
            outputs = ret.last_hidden_state if hasattr(ret, 'last_hidden_state') else ret.hidden_states[-1]
        else:
            ret = self.model(output_hidden_states=True, return_dict=True, **inputs)
            all_layer_outputs = list(ret.hidden_states)
            if hasattr(ret, 'last_hidden_state'):
                all_layer_outputs[-1] = ret.last_hidden_state
            if return_all_layer_outputs:
                return (all_layer_outputs, ret.logits) if return_mlm_logits else all_layer_outputs
            outputs = all_layer_outputs[layer_index]
        outputs = get_pooling(outputs, inputs,
                              pooling_strategy or self.pooling_strategy,
                              padding_side=self.padding_side)
        n_dim = len(outputs.shape)
        if embedding_start is not None:
            if n_dim == 2:
                outputs = outputs[:, embedding_start:]
            elif n_dim == 3:
                outputs = outputs[:, :, embedding_start:]
            else:
                raise ValueError(f'Unsupported output shape: {outputs.shape}')
        if embedding_size is not None:
            if n_dim == 2:
                outputs = outputs[:, :embedding_size]
            elif n_dim == 3:
                outputs = outputs[:, :, :embedding_size]
            else:
                raise ValueError(f'Unsupported output shape: {outputs.shape}')
        return (outputs, ret.logits) if return_mlm_logits else outputs

# ============================================================================
# 5. 数据整理器
# ============================================================================
def detect_dataset_format(ds) -> str:
    if hasattr(ds, '__getitem__') and len(ds) > 0:
        sample = ds[0]
        if 'text1' in sample and 'text2' in sample and 'label' in sample:
            return 'A'
        elif 'query' in sample and 'positive' in sample and 'negative' in sample:
            return 'C'
        elif 'query' in sample and 'positive' in sample:
            return 'B'
    return 'A'

@dataclass
class AngleDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = 'longest'
    max_length: Optional[int] = None
    return_tensors: str = "pt"
    filter_duplicate: bool = True
    text_prompt: Optional[str] = None
    query_prompt: Optional[str] = None
    doc_prompt: Optional[str] = None
    dataset_format: Optional[str] = None

    @staticmethod
    def sample_from_list(text: Union[str, List[str]]) -> str:
        if isinstance(text, list):
            return random.choice(text)
        return text

    def __call__(self, features: List[Dict], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        if self.dataset_format is None:
            sample = features[0]
            if 'text1' in sample and 'text2' in sample and 'label' in sample:
                self.dataset_format = 'A'
                logger.info('检测到数据集格式 A: text1, text2, label')
            elif 'query' in sample and 'positive' in sample and 'negative' in sample:
                self.dataset_format = 'C'
                logger.info('检测到数据集格式 C: query, positive, negative')
            elif 'query' in sample and 'positive' in sample and 'negative' not in sample:
                self.dataset_format = 'B'
                logger.info('检测到数据集格式 B: query, positive')
            else:
                raise NotImplementedError('只支持格式A/B/C')
        processed_features = []
        duplicate_set = set()
        for feature in features:
            texts = []
            label = -1
            if self.dataset_format == 'A':
                text1 = self.sample_from_list(feature['text1'])
                text2 = self.sample_from_list(feature['text2'])
                label = float(feature['label'])
                if self.text_prompt is not None:
                    text1 = self.text_prompt.format(text=text1)
                    text2 = self.text_prompt.format(text=text2)
                texts = [text1, text2]
            elif self.dataset_format == 'B':
                query = self.sample_from_list(feature['query'])
                positive = self.sample_from_list(feature['positive'])
                if self.query_prompt is not None:
                    query = self.query_prompt.format(text=query)
                if self.doc_prompt is not None:
                    positive = self.doc_prompt.format(text=positive)
                texts = [query, positive]
            elif self.dataset_format == 'C':
                query = self.sample_from_list(feature['query'])
                positive = self.sample_from_list(feature['positive'])
                negative = self.sample_from_list(feature['negative'])
                if self.query_prompt is not None:
                    query = self.query_prompt.format(text=query)
                if self.doc_prompt is not None:
                    positive = self.doc_prompt.format(text=positive)
                    negative = self.doc_prompt.format(text=negative)
                texts = [query, positive, negative]
            tokenized_texts = []
            is_duplicate = False
            for text in texts:
                tok = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    add_special_tokens=True
                )
                input_ids_tuple = tuple(tok['input_ids'])
                if self.filter_duplicate and input_ids_tuple in duplicate_set:
                    is_duplicate = True
                    break
                duplicate_set.add(input_ids_tuple)
                tok['labels'] = [label]
                tokenized_texts.append(tok)
            if self.filter_duplicate and is_duplicate:
                continue
            processed_features.extend(tokenized_texts)
        if not processed_features:
            raise ValueError('没有特征可处理，请考虑禁用filter_duplicate')
        batch = self.tokenizer.pad(
            {'input_ids': [f['input_ids'] for f in processed_features]},
            padding=self.padding,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        batch['labels'] = torch.Tensor([f['labels'] for f in processed_features])
        return batch

# ============================================================================
# 6. AngleTrainer类
# ============================================================================
class AngleTrainer(Trainer):
    def __init__(self,
                 pooler: Pooler,
                 loss_kwargs: Optional[Dict] = None,
                 dataset_format: str = 'A',
                 teacher_name_or_path: Optional[str] = None,
                 teacher_pooling_strategy: str = 'cls',
                 pad_token_id: int = 0,
                 model_kwargs: Optional[Dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pooler = pooler
        self.pad_token_id = pad_token_id
        self.model_kwargs = model_kwargs
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_fct = AngleLoss(dataset_format=dataset_format, **loss_kwargs)
        self.teacher_name_or_path = teacher_name_or_path
        self.teacher_pooling_strategy = teacher_pooling_strategy
        if teacher_name_or_path is not None:
            logger.info('检测到教师模型! 请确保教师模型与骨干模型有相同的分词器!')
            teacher_backbone = AutoModel.from_pretrained(
                teacher_name_or_path,
                trust_remote_code=True,
                torch_dtype=self.pooler.model.dtype,
                **self.model_kwargs).to(self.pooler.model.device)
            self.teacher_pooler = Pooler(
                teacher_backbone,
                pooling_strategy=self.teacher_pooling_strategy,
                padding_side=self.pooler.padding_side)
            logger.info(f'使用教师模型训练: {teacher_name_or_path}')

    def compute_distillation_loss(self,
                                  inputs: torch.Tensor,
                                  targets: torch.Tensor,
                                  mse_weight: float = 1.0,
                                  kl_temperature: float = 1.0) -> torch.Tensor:
        loss = 0.
        if mse_weight > 0:
            loss += mse_weight * nn.MSELoss()(inputs, targets)
        if kl_temperature > 0:
            loss += nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(inputs / kl_temperature, dim=-1),
                F.softmax(targets / kl_temperature, dim=-1)
            ) * kl_temperature
        return loss

    def compute_mlm_loss(self, logits, mask_target_labels):
        return F.cross_entropy(
            logits.transpose(1, 2),
            mask_target_labels,
            ignore_index=self.pad_token_id,
        )

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels", None)
        mask_target_labels = inputs.pop("mask_target_labels", None)
        if mask_target_labels is not None:
            all_layer_outputs, mlm_logits = self.pooler(
                inputs, layer_index=-1, return_all_layer_outputs=True, return_mlm_logits=True)
        else:
            all_layer_outputs = self.pooler(inputs, layer_index=-1, return_all_layer_outputs=True)
        all_outputs = all_layer_outputs[-1]
        outputs = get_pooling(all_outputs, inputs,
                              self.pooler.pooling_strategy,
                              self.pooler.padding_side)
        actual_model = model.module if hasattr(model, 'module') else model
        if hasattr(actual_model, 'complex_proj'):
            outputs = actual_model.complex_proj(outputs)
        loss = self.loss_fct(labels, outputs)
        if self.teacher_name_or_path is not None:
            with torch.no_grad():
                self.teacher_pooler.model = self.teacher_pooler.model.to(self.pooler.model.device)
                align_outputs = self.teacher_pooler(inputs)
            alignment_loss = self.compute_distillation_loss(
                all_outputs if self.teacher_pooling_strategy == 'all' else outputs,
                align_outputs,
                mse_weight=0.0,
                kl_temperature=1.0)
            loss += alignment_loss
        if mask_target_labels is not None:
            loss += self.compute_mlm_loss(mlm_logits, mask_target_labels)
        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, *args, **kwargs):
        eval_loss = self.compute_loss(model, inputs, return_outputs=False)
        return eval_loss, None, None

# ============================================================================
# 7. AnglE主类
# ============================================================================
class AngleBase:
    pass

class AnglE(AngleBase):
    cfg_file_name = 'angle_config.json'
    special_columns = ['labels']

    def __init__(self,
                 model_name_or_path: str,
                 tokenizer_name_or_path: Optional[str] = None,
                 max_length: int = 512,
                 model_kwargs: Optional[Dict] = None,
                 pooling_strategy: Optional[str] = None,
                 train_mode: bool = True,
                 torch_dtype: Optional[torch.dtype] = None,
                 device: Optional[str] = None,
                 tokenizer_padding_side: Optional[str] = None,
                 **kwargs: Any):
        super().__init__()
        self.max_length = max_length
        self.train_mode = train_mode
        self.pooling_strategy = pooling_strategy or 'cls'
        if device is not None:
            self.device = device
        else:
            self.device = set_device()
        if torch_dtype is None:
            torch_dtype = torch.float32 if train_mode else None
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or model_name_or_path, trust_remote_code=True)
        if tokenizer_padding_side is not None and self.tokenizer.padding_side != tokenizer_padding_side:
            self.tokenizer.padding_side = tokenizer_padding_side
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            **self.model_kwargs)
        hidden_size = self.model.config.hidden_size
        self.model.add_module('complex_proj', ComplexProjection(hidden_size))
        proj_path = os.path.join(model_name_or_path, 'complex_proj.bin')
        if os.path.exists(proj_path):
            self.model.complex_proj.load_state_dict(torch.load(proj_path, map_location='cpu'))
            logger.info(f"✅ 成功加载 AP-Proj 复数投影层权重: {proj_path}")
        else:
            logger.info("🆕 初始化了全新的 AP-Proj 复数投影层")
        if train_mode:
            self.model.train()
        else:
            self.model.eval()
        self.model.config.use_cache = False
        self.pooler = Pooler(
            self.model,
            pooling_strategy=self.pooling_strategy,
            padding_side=self.tokenizer.padding_side)
        self.__cfg = {
            'model_name_or_path': model_name_or_path,
            'max_length': max_length,
            'model_kwargs': model_kwargs,
            'pooling_strategy': self.pooling_strategy,
            'angle_emb_version': '0.0.1',
        }
        self.__cfg.update(kwargs)

    def cuda(self):
        self.model = self.model.to(torch.device(self.device))
        return self

    def to(self, device: Any):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = self.model.to(device)
        self.device = device
        return self

    @staticmethod
    def from_pretrained(model_name_or_path: str,
                        max_length: int = 512,
                        pooling_strategy: str = 'cls',
                        train_mode: bool = False,
                        model_kwargs: Optional[Dict] = None,
                        **kwargs):
        kwargs['model_kwargs'] = model_kwargs
        angle = AnglE(model_name_or_path,
                      max_length=max_length,
                      pooling_strategy=pooling_strategy,
                      train_mode=train_mode,
                      **kwargs)
        return angle

    def save_config(self, fpath: str):
        with open(fpath, 'w', encoding='utf-8') as writer:
            json.dump(self.__cfg, writer, ensure_ascii=False, indent=2)

    def fit(self,
            train_ds,
            valid_ds=None,
            batch_size: int = 32,
            output_dir: Optional[str] = None,
            epochs: int = 1,
            learning_rate: float = 1e-5,
            warmup_steps: int = 1000,
            logging_steps: int = 10,
            eval_steps: int = 1000,
            eval_strategy: str = 'steps',
            save_steps: int = 100,
            save_strategy: str = 'steps',
            save_total_limit: int = 1,
            gradient_accumulation_steps: int = 1,
            fp16: Optional[bool] = None,
            bf16: Optional[bool] = None,
            argument_kwargs: Optional[Dict] = None,
            trainer_kwargs: Optional[Dict] = None,
            loss_kwargs: Optional[Dict] = None,
            filter_duplicate: bool = True,
            padding: str = 'longest',
            text_prompt: Optional[str] = None,
            query_prompt: Optional[str] = None,
            doc_prompt: Optional[str] = None):
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        self.save_config(os.path.join(output_dir, AnglE.cfg_file_name))
        self.tokenizer.save_pretrained(output_dir)
        if fp16 is None:
            fp16 = False
        if bf16 is None:
            bf16 = False
        if argument_kwargs is None:
            argument_kwargs = {}
        if trainer_kwargs is None:
            trainer_kwargs = {}
        dataset_format = detect_dataset_format(train_ds)
        logger.info(f'数据集格式: {dataset_format}')
        if not isinstance(train_ds, list):
            train_ds = list(train_ds)
        if valid_ds is not None and not isinstance(valid_ds, list):
            valid_ds = list(valid_ds)
        trainer = AngleTrainer(
            pooler=self.pooler,
            model=self.model,
            dataset_format=dataset_format,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            loss_kwargs=loss_kwargs,
            tokenizer=self.tokenizer,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                fp16=fp16,
                bf16=bf16,
                logging_steps=logging_steps,
                save_steps=save_steps,
                save_strategy=save_strategy,
                eval_strategy=eval_strategy if valid_ds is not None else 'no',
                eval_steps=eval_steps,
                output_dir=output_dir,
                save_total_limit=save_total_limit,
                load_best_model_at_end=False,
                ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
                remove_unused_columns=False,
                **argument_kwargs,
            ),
            data_collator=AngleDataCollator(
                self.tokenizer,
                padding=padding,
                return_tensors="pt",
                max_length=self.max_length,
                filter_duplicate=filter_duplicate,
                text_prompt=text_prompt,
                query_prompt=query_prompt,
                doc_prompt=doc_prompt,
                dataset_format=dataset_format,
            ),
            model_kwargs=self.model_kwargs,
            **trainer_kwargs
        )
        trainer.train()
        self.save_pretrained(output_dir)
        logger.info(f"✅ 模型训练完成并保存到: {output_dir}")

    @torch.no_grad()
    def encode(self,
               inputs: Union[List[str], Tuple[str], str],
               max_length: Optional[int] = None,
               to_numpy: bool = True,
               device: Optional[Any] = None,
               prompt: Optional[str] = None,
               normalize_embedding: bool = False,
               padding: str = 'longest'):
        self.model.eval()
        device = device or self.model.device
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        if prompt is not None:
            inputs = [prompt.format(text=text) for text in inputs]
        tok = self.tokenizer(
            inputs,
            padding=padding,
            max_length=max_length or self.max_length,
            truncation=True,
            return_tensors='pt')
        tok.to(device)
        output = self.pooler(tok)
        if hasattr(self.model, 'complex_proj'):
            output = self.model.complex_proj(output)
        if normalize_embedding:
            output = nn.functional.normalize(output, p=2, dim=-1)
        if to_numpy:
            return output.float().detach().cpu().numpy()
        return output

    def save_pretrained(self, output_dir: str, exist_ok: bool = True):
        if not exist_ok and os.path.exists(output_dir):
            raise ValueError(f"输出目录 ({output_dir}) 已存在且不为空")
        os.makedirs(output_dir, exist_ok=exist_ok)
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
        if hasattr(self.model, 'complex_proj'):
            torch.save(self.model.complex_proj.state_dict(), os.path.join(output_dir, 'complex_proj.bin'))
            logger.info(f"💾 AP-Proj 投影层权重已保存至: {os.path.join(output_dir, 'complex_proj.bin')}")

# ============================================================================
# 8. 训练器包装类（支持消融实验）
# ============================================================================
class AngleTrainerWrapper:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ 配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise

    def save_config(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        config_save_path = os.path.join(output_dir, "train_config.json")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 配置已保存至: {config_save_path}")

    def cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
            logger.info(f"🧹 内存清理完成 - 已分配: {allocated:.2f}GB, 可用: {free:.2f}GB")

    def print_gpu_info(self):
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory = gpu_props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            free = total_memory - allocated
            logger.info(f"🎮 GPU: {gpu_props.name}")
            logger.info(f"💾 显存: 总共 {total_memory:.1f}GB, 已用 {allocated:.2f}GB, 可用 {free:.2f}GB")
            logger.info(f"🔧 CUDA能力: {gpu_props.major}.{gpu_props.minor}")
        else:
            logger.warning("⚠️ 未检测到CUDA设备，将使用CPU训练")

    def load_and_prepare_data_raw(self) -> List[Dict]:
        """
        从原始 CSV 文件加载数据，生成完整评论的三元组（无裁剪）
        返回格式：每个样本 {'text1': full_review, 'text2': aspect_name, 'label': polarity}
        """
        csv_file = self.config['data'].get('original_csv_file')
        if not csv_file:
            raise ValueError("未配置 original_csv_file，无法启用 disable_dynamic_window")

        if not os.path.exists(csv_file):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_path = os.path.join(current_dir, csv_file)
            if os.path.exists(possible_path):
                csv_file = possible_path
            else:
                raise FileNotFoundError(f"原始 CSV 文件不存在: {csv_file}")

        logger.info(f"📥 [RAW MODE] 正在加载原始 CSV 文件（无裁剪）: {csv_file}")

        dataset = []
        polarity_map = {
            -2: 0, -1: 0,
            0: 1,
            1: 2, 2: 2
        }

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # 获取所有 aspect 列名（排除 id, review, star）
            fieldnames = reader.fieldnames
            aspect_columns = [col for col in fieldnames if col not in ['id', 'review', 'star']]

            for row in reader:
                review = row['review'].strip()
                if not review:
                    continue
                for col in aspect_columns:
                    val_str = row.get(col)
                    if val_str is None or val_str == '':
                        continue
                    try:
                        val = int(val_str)
                    except ValueError:
                        continue
                    if val < -2 or val > 2:
                        continue
                    label = polarity_map.get(val)
                    if label is None:
                        continue
                    aspect_name = col.replace('#', ' ')
                    dataset.append({
                        'text1': review,
                        'text2': aspect_name,
                        'label': label
                    })

        logger.info(f"✅ 成功加载 {len(dataset)} 条原始数据（无裁剪）")
        if len(dataset) > 0:
            logger.info("🔍 原始数据样本示例:")
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                logger.info(f"  样本 {i+1}:")
                logger.info(f"    text1: {sample['text1'][:80]}...")
                logger.info(f"    text2: {sample['text2']}")
                logger.info(f"    label: {sample['label']}")
        return dataset

    def load_and_prepare_data(self):
        """原有数据加载方法，可能包含裁剪"""
        if TripletDataConverter is None:
            raise ImportError("TripletDataConverter 未找到，无法使用原有数据加载方法")
        input_file = self.config['data']['input_jsonl_file']
        logger.info(f"📥 正在加载数据集: {input_file}")
        if not os.path.exists(input_file):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_path = os.path.join(current_dir, input_file)
            if os.path.exists(possible_path):
                input_file = possible_path
            else:
                raise FileNotFoundError(f"数据文件不存在: {input_file}")
        dataset = TripletDataConverter.load_and_convert(input_file)
        if len(dataset) > 0:
            logger.info("🔍 数据样本示例:")
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                logger.info(f"  样本 {i+1}:")
                logger.info(f"    text1: {sample['text1'][:80]}...")
                logger.info(f"    text2: {sample['text2'][:80]}...")
                logger.info(f"    label: {sample['label']}")
        return dataset

    def train(self):
        logger.info("🚀 开始AnglE模型训练")
        # 显示配置
        logger.info("📋 训练配置:")
        logger.info(f"  模型: {self.config['model']['name']}")
        logger.info(f"  Batch Size: {self.config['training']['batch_size']}")
        logger.info(f"  梯度累积: {self.config['training']['gradient_accumulation_steps']}")
        logger.info(f"  有效Batch Size: {self.config['training']['batch_size'] * self.config['training']['gradient_accumulation_steps']}")
        logger.info(f"  序列长度: {self.config['model']['max_length']}")
        logger.info(f"  学习率: {self.config['training']['learning_rate']}")
        logger.info(f"  训练轮数: {self.config['training']['num_epochs']}")

        disable_dynamic_window = self.config.get('data', {}).get('disable_dynamic_window', False)
        if disable_dynamic_window:
            logger.info("🔧 消融实验模式: 禁用动态窗口 - 使用完整原始文本（无裁剪）")
        else:
            logger.info("🔧 正常模式: 使用动态窗口裁剪")

        self.cleanup_memory()
        self.print_gpu_info()

        # 加载模型
        try:
            logger.info("🤖 正在加载模型...")
            angle = AnglE.from_pretrained(
                model_name_or_path=self.config['model']['name'],
                max_length=self.config['model']['max_length'],
                pooling_strategy=self.config['model']['pooling_strategy'],
                train_mode=True
            )
            if torch.cuda.is_available():
                angle = angle.cuda()
                logger.info("✅ 模型已转移到GPU")
            else:
                logger.warning("⚠️ 使用CPU进行训练，速度会较慢")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return

        # 加载数据
        try:
            if disable_dynamic_window:
                train_ds = self.load_and_prepare_data_raw()
            else:
                train_ds = self.load_and_prepare_data()
            logger.info(f"✅ 成功加载 {len(train_ds)} 条训练数据")
            if len(train_ds) == 0:
                logger.error("❌ 没有训练数据！请检查数据文件")
                return
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            return

        output_dir = self.config['data']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self.save_config(output_dir)

        logger.info("🎯 开始模型训练...")
        try:
            fit_args = {
                'train_ds': train_ds,
                'output_dir': output_dir,
                'batch_size': self.config['training']['batch_size'],
                'gradient_accumulation_steps': self.config['training']['gradient_accumulation_steps'],
                'epochs': self.config['training']['num_epochs'],
                'learning_rate': self.config['training']['learning_rate'],
                'save_steps': self.config['training']['save_steps'],
                'warmup_steps': self.config['training']['warmup_steps'],
                'loss_kwargs': self.config['loss'],
                'logging_steps': self.config['training']['logging_steps'],
            }
            if self.config['training']['fp16'] and torch.cuda.is_available():
                fit_args['fp16'] = True
                logger.info("🔧 启用FP16混合精度训练")
            else:
                fit_args['fp16'] = False
                logger.info("🔧 禁用FP16")
            angle.fit(**fit_args)
            logger.info(f"🎉 训练完成！模型已保存至: {output_dir}")
        except torch.cuda.OutOfMemoryError:
            logger.error("💥 GPU内存不足！")
            self.handle_oom_error()
        except Exception as e:
            logger.error(f"💥 训练过程中出现错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if self.config['training']['fp16']:
                logger.info("🔄 尝试禁用FP16...")
                try:
                    fit_args['fp16'] = False
                    angle.fit(**fit_args)
                    logger.info(f"🎉 训练完成！模型已保存至: {output_dir}")
                except Exception as e2:
                    logger.error(f"💥 重试也失败了: {e2}")
                    raise
            else:
                raise

    def handle_oom_error(self):
        logger.info("💡 内存优化建议:")
        logger.info("  1. 降低 batch_size")
        logger.info("  2. 降低 max_length")
        logger.info("  3. 禁用 FP16")
        logger.info("  4. 使用更小的模型")

def main():
    # 使用包含 disable_dynamic_window 的配置文件
    config_path = r"Ablation_eperience\train\dynamic_window\config\train_config.json"
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return
    trainer = AngleTrainerWrapper(config_path)
    trainer.train()

if __name__ == '__main__':
    main()