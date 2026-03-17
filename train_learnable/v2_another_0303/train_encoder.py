import torch
import os
import sys
import gc
import json
import logging
import random
import math
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

# 使用相对路径导入data_converter
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, 'utils')
sys.path.insert(0, utils_path)

from data_converter import TripletDataConverter

# ============================================================================
# 1. 工具函数 (从utils.py)
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
    elif isinstance(pooling_strategy, int) or pooling_strategy.isnumeric():
        # index
        outputs = outputs[:, int(pooling_strategy)]
    else:
        raise NotImplementedError(
            'please specify pooling_strategy from '
            '[`cls`, `cls_avg`, `cls_max`, `last`, `avg`, `mean`, `max`, `all`, int]')
    return outputs

# ============================================================================
# 1.5 自定义复数空间映射层 (方案1: AP-Proj)
# ============================================================================
class ComplexProjection(nn.Module):
    """可学习的复数空间投影层 (Aspect-Polarity Projection)"""
    def __init__(self, hidden_size):
        super().__init__()
        # 将输入维度隐式映射到复数空间的实部(Aspect)和虚部(Polarity)
        self.proj_re = nn.Linear(hidden_size, hidden_size // 2)
        self.proj_im = nn.Linear(hidden_size, hidden_size // 2)
        
        # 初始化权重 (帮助模型在初期能稳定收敛)
        nn.init.xavier_uniform_(self.proj_re.weight)
        nn.init.xavier_uniform_(self.proj_im.weight)
        nn.init.zeros_(self.proj_re.bias)
        nn.init.zeros_(self.proj_im.bias)
        
    def forward(self, x):
        re = self.proj_re(x)
        im = self.proj_im(x)
        # 拼接输出，保持原有的总维度不变，以便下游 torch.chunk 完美衔接
        return torch.cat([re, im], dim=1)

# ============================================================================
# 2. 损失函数 (从loss.py完整复制)
# ============================================================================

# ============================================================================
# 2. 损失函数 (从loss.py完整复制)
# ============================================================================

def categorical_crossentropy_loss(y_true: torch.Tensor, y_pred: torch.Tensor, from_logits: bool = True) -> torch.Tensor:
    """计算分类交叉熵"""
    if from_logits:
        return -(F.log_softmax(y_pred, dim=1) * y_true).sum(dim=1)
    return -(torch.log(y_pred, dim=1) * y_true).sum(dim=1)


def cosine_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    """计算余弦损失"""
    # y_true必须为交错格式: [x[0][0], x[0][1], x[1][0], x[1][1], ...]
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
    """计算角度损失"""
    y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()

    y_pred_re, y_pred_im = torch.chunk(y_pred, 2, dim=1)
    a = y_pred_re[::2]
    b = y_pred_im[::2]
    c = y_pred_re[1::2]
    d = y_pred_im[1::2]

    # (a+bi) / (c+di)
    # = ((a+bi) * (c-di)) / ((c+di) * (c-di))
    # = ((ac + bd) + i(bc - ad)) / (c^2 + d^2)
    # = (ac + bd) / (c^2 + d^2) + i(bc - ad)/(c^2 + d^2)
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
    y_pred = torch.abs(pooling) * tau  # absolute delta angle
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
    """计算批内负样本损失"""
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

    # compute similarity
    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = y_pred @ y_pred.T  # dot product
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
    """计算带负样本的对比损失"""
    target = torch.cat((pos, neg), dim=0) if neg is not None else pos  # (2B, D)
    q_norm = torch.nn.functional.normalize(text, p=2, dim=1)  # (B, D)
    t_norm = torch.nn.functional.normalize(target, p=2, dim=1)  # (2B, D)
    scores = torch.mm(q_norm, t_norm.transpose(0, 1)) * tau  # (B, 2B)
    labels = torch.tensor(
        range(len(scores)), dtype=torch.long, device=scores.device
    )
    return nn.CrossEntropyLoss()(scores, labels)


class AngleLoss:
    """AnglE损失函数配置"""
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
        """计算AnglE损失"""
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
            # 格式 B: query, positive (无负样本)
            query = outputs[::2]
            positive = outputs[1::2]
            loss = contrastive_with_negative_loss(query, positive, neg=None, tau=self.ibn_tau)
            
        elif self.dataset_format == 'C':
            # 格式 C: query, positive, negative
            if int(self.cln_w) == 0:
                logger.info('`cln_w` is set to zero. Contrastive learning with hard negative is disabled. '
                            'Please manually check whether it is correct.')
            
            query = outputs[::3]
            positive = outputs[1::3]
            negative = outputs[2::3]
            assert query.shape == positive.shape == negative.shape, f'query.shape={query.shape}, positive.shape={positive.shape}, negative.shape={negative.shape}'  # NOQA

            _, fea_dim = query.shape
            positive_inputs = torch.stack((query, positive), dim=1).reshape(-1, fea_dim)  # zip(query, positive)
            positive_labels = torch.ones_like(positive_inputs[:, :1]).long()
            negative_inputs = torch.stack((query, negative), dim=1).reshape(-1, fea_dim)  # zip(query, negative)
            negative_labels = torch.zeros_like(negative_inputs[:, :1]).long()
            combined_inputs = torch.cat((positive_inputs, negative_inputs), dim=0)
            combined_labels = torch.cat((positive_labels, negative_labels), dim=0)

            loss = 0.
            # contrastive learning loss
            cll = 0.
            if self.ibn_w > 0:
                cll += self.ibn_w * contrastive_with_negative_loss(query, positive, tau=self.ibn_tau)
            if self.cln_w > 0:
                cll += self.cln_w * contrastive_with_negative_loss(query, positive, negative, tau=self.ibn_tau)
            if cll > 0:
                loss += cll / 2
            # angle loss
            if self.angle_w > 0:
                loss += self.angle_w * angle_loss(combined_labels, combined_inputs, self.angle_tau,
                                                  pooling_strategy=self.angle_pooling_strategy)
            # cosine loss
            if self.cosine_w > 0:
                loss += self.cosine_w * cosine_loss(combined_labels, combined_inputs, self.cosine_tau)
        else:
            raise NotImplementedError
        return loss


# ============================================================================
# 3. Pooler类 (从angle.py)
# ============================================================================

class Pooler:
    """使用Pooler获取句子嵌入"""
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
        """获取句子嵌入"""
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
            # topk embedding size
            if n_dim == 2:
                outputs = outputs[:, :embedding_size]
            elif n_dim == 3:
                outputs = outputs[:, :, :embedding_size]
            else:
                raise ValueError(f'Unsupported output shape: {outputs.shape}')
        return (outputs, ret.logits) if return_mlm_logits else outputs


# ============================================================================
# 4. 数据整理器 (从angle.py完整复制)
# ============================================================================

def detect_dataset_format(ds) -> str:
    """从原始数据检测数据集格式"""
    if hasattr(ds, '__getitem__') and len(ds) > 0:
        sample = ds[0]
        if 'text1' in sample and 'text2' in sample and 'label' in sample:
            return 'A'
        elif 'query' in sample and 'positive' in sample and 'negative' in sample:
            return 'C'
        elif 'query' in sample and 'positive' in sample:
            return 'B'
    return 'A'  # 默认格式A


@dataclass
class AngleDataCollator:
    """AngleDataCollator，处理原始数据，分词并准备批次"""
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
        """从列表中选择一个字符串或直接返回字符串"""
        if isinstance(text, list):
            return random.choice(text)
        return text

    def __call__(self, features: List[Dict], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """整理函数，处理原始数据"""
        if return_tensors is None:
            return_tensors = self.return_tensors

        # 从第一个样本自动检测数据集格式
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
                raise NotImplementedError(
                    '目前只支持三种数据集格式: '
                    '格式 A: 必须包含三列: `text1`, `text2`, `label`. '
                    '格式 B: 必须包含两列: `query`, `positive`. '
                    '格式 C: 必须包含三列: `query`, `positive`, `negative`.'
                )
        
        # 根据格式处理特征
        processed_features = []
        duplicate_set = set()

        for feature in features:
            texts = []
            label = -1

            if self.dataset_format == 'A':
                # 格式 A: text1, text2, label
                text1 = self.sample_from_list(feature['text1'])
                text2 = self.sample_from_list(feature['text2'])
                label = float(feature['label'])

                # 如果提供了text_prompt，应用它（仅格式A）
                if self.text_prompt is not None:
                    text1 = self.text_prompt.format(text=text1)
                    text2 = self.text_prompt.format(text=text2)

                texts = [text1, text2]

            elif self.dataset_format == 'B':
                # 格式 B: query, positive
                query = self.sample_from_list(feature['query'])
                positive = self.sample_from_list(feature['positive'])

                # 应用提示
                if self.query_prompt is not None:
                    query = self.query_prompt.format(text=query)

                if self.doc_prompt is not None:
                    positive = self.doc_prompt.format(text=positive)

                texts = [query, positive]

            elif self.dataset_format == 'C':
                # 格式 C: query, positive, negative
                query = self.sample_from_list(feature['query'])
                positive = self.sample_from_list(feature['positive'])
                negative = self.sample_from_list(feature['negative'])

                # 应用提示
                if self.query_prompt is not None:
                    query = self.query_prompt.format(text=query)

                if self.doc_prompt is not None:
                    positive = self.doc_prompt.format(text=positive)
                    negative = self.doc_prompt.format(text=negative)

                texts = [query, positive, negative]

            # 分词
            tokenized_texts = []
            is_duplicate = False
            for text in texts:
                tok = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    add_special_tokens=True
                )

                # 检查重复
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

        # 填充并转换为张量
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
# 5. AngleTrainer类 (从angle.py完整复制)
# ============================================================================

class AngleTrainer(Trainer):
    """自定义Huggingface Trainer用于AnglE"""
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
        """计算蒸馏损失"""
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
        """计算AnglE的损失"""
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
        
        # ======================= [在这里插入新增代码] =======================
        # 处理 DDP/DataParallel 包装模型的情况
        actual_model = model.module if hasattr(model, 'module') else model
        if hasattr(actual_model, 'complex_proj'):
            outputs = actual_model.complex_proj(outputs)
        # ====================================================================

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
# 6. AnglE主类 (从angle.py简化但保持核心功能)
# ============================================================================

class AngleBase:
    """AnglE基类"""
    pass


class AnglE(AngleBase):
    """AnglE主类"""
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

        # 加载模型
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            **self.model_kwargs)
        
        # ======================= [在这里插入新增代码] =======================
        hidden_size = self.model.config.hidden_size
        self.model.add_module('complex_proj', ComplexProjection(hidden_size))
        
        # 尝试加载可能已经训练好的投影层权重（用于后续的推理和评估）
        proj_path = os.path.join(model_name_or_path, 'complex_proj.bin')
        if os.path.exists(proj_path):
            self.model.complex_proj.load_state_dict(torch.load(proj_path, map_location='cpu'))
            logger.info(f"✅ 成功加载 AP-Proj 复数投影层权重: {proj_path}")
        else:
            logger.info("🆕 初始化了全新的 AP-Proj 复数投影层")
        # ====================================================================


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
        """从预训练模型加载AnglE"""
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
        """
        使用AnglE进行训练
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        # 保存配置
        self.save_config(os.path.join(output_dir, AnglE.cfg_file_name))
        # 保存分词器
        self.tokenizer.save_pretrained(output_dir)

        if fp16 is None:
            fp16 = False
        if bf16 is None:
            bf16 = False

        # 初始化argument_kwargs
        if argument_kwargs is None:
            argument_kwargs = {}

        if trainer_kwargs is None:
            trainer_kwargs = {}

        # 检测数据集格式
        dataset_format = detect_dataset_format(train_ds)
        logger.info(f'数据集格式: {dataset_format}')
        
        # 将train_ds转换为列表格式（如果需要）
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

        # trainer.train()
        # self.model.save_pretrained(output_dir)
        # logger.info(f"✅ 模型训练完成并保存到: {output_dir}")

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
        """
        编码文本
        """
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

        # ======================= [在这里插入新增代码] =======================
        if hasattr(self.model, 'complex_proj'):
            output = self.model.complex_proj(output)
        # ===================================================================

        if normalize_embedding:
            output = nn.functional.normalize(output, p=2, dim=-1)
        if to_numpy:
            return output.float().detach().cpu().numpy()
        return output

    def save_pretrained(self, output_dir: str, exist_ok: bool = True):
        """保存模型和分词器"""
        if not exist_ok and os.path.exists(output_dir):
            raise ValueError(f"输出目录 ({output_dir}) 已存在且不为空")
        os.makedirs(output_dir, exist_ok=exist_ok)
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)

        # ======================= [在这里插入新增代码] =======================
        if hasattr(self.model, 'complex_proj'):
            torch.save(self.model.complex_proj.state_dict(), os.path.join(output_dir, 'complex_proj.bin'))
            logger.info(f"💾 AP-Proj 投影层权重已保存至: {os.path.join(output_dir, 'complex_proj.bin')}")
        # ====================================================================


# ============================================================================
# 7. 训练器主类
# ============================================================================

class AngleTrainerWrapper:
    def __init__(self, config_path=r"train_learnable\v2_another_0303\config\train_config.json"):
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ 配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise
    
    def save_config(self, output_dir):
        """保存配置到输出目录"""
        os.makedirs(output_dir, exist_ok=True)
        config_save_path = os.path.join(output_dir, "train_config.json")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 配置已保存至: {config_save_path}")
    
    def cleanup_memory(self):
        """清理GPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
            logger.info(f"🧹 内存清理完成 - 已分配: {allocated:.2f}GB, 可用: {free:.2f}GB")
    
    def print_gpu_info(self):
        """打印GPU信息"""
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
    
    def load_and_prepare_data(self):
        """加载和准备数据 - 使用外部的data_converter"""
        try:
            input_file = self.config['data']['input_jsonl_file']
            logger.info(f"📥 正在加载数据集: {input_file}")
            
            # 检查文件是否存在
            if not os.path.exists(input_file):
                logger.error(f"❌ 数据文件不存在: {input_file}")
                # 尝试使用相对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                possible_path = os.path.join(current_dir, input_file)
                if os.path.exists(possible_path):
                    input_file = possible_path
                    logger.info(f"📁 使用相对路径: {input_file}")
                else:
                    raise FileNotFoundError(f"数据文件不存在: {input_file}")
            
            # 使用数据转换器加载和转换数据
            dataset = TripletDataConverter.load_and_convert(input_file)
            
            # 显示样本示例
            if len(dataset) > 0:
                logger.info("🔍 数据样本示例:")
                for i in range(min(3, len(dataset))):
                    sample = dataset[i]
                    logger.info(f"  样本 {i+1}:")
                    logger.info(f"    text1: {sample['text1'][:80]}...")
                    logger.info(f"    text2: {sample['text2'][:80]}...")
                    logger.info(f"    label: {sample['label']}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def train(self):
        """主训练函数"""
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
        
        # 清理内存并显示GPU信息
        self.cleanup_memory()
        self.print_gpu_info()
        
        # 1. 加载模型
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
        
        # 2. 加载数据
        try:
            train_ds = self.load_and_prepare_data()
            logger.info(f"✅ 成功加载 {len(train_ds)} 条训练数据")
            
            if len(train_ds) == 0:
                logger.error("❌ 没有训练数据！请检查数据文件")
                return
                
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            return
        
        # 3. 保存配置
        output_dir = self.config['data']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        self.save_config(output_dir)
        
        # 4. 开始训练
        logger.info("🎯 开始模型训练...")
        try:
            # 基础训练参数
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
            
            # FP16配置
            if self.config['training']['fp16'] and torch.cuda.is_available():
                fit_args['fp16'] = True
                logger.info("🔧 启用FP16混合精度训练")
            else:
                fit_args['fp16'] = False
                logger.info("🔧 禁用FP16")
            
            # 开始训练
            angle.fit(**fit_args)
            
            logger.info(f"🎉 训练完成！模型已保存至: {output_dir}")
            
        except torch.cuda.OutOfMemoryError:
            logger.error("💥 GPU内存不足！")
            self.handle_oom_error()
            
        except Exception as e:
            logger.error(f"💥 训练过程中出现错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 尝试不使用FP16
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
        """处理内存不足错误"""
        logger.info("💡 内存优化建议:")
        logger.info("  1. 降低 batch_size")
        logger.info("  2. 降低 max_length")
        logger.info("  3. 禁用 FP16")
        logger.info("  4. 使用更小的模型")


def main():
    """主函数"""
    try:
        # 可以指定不同的配置文件
        config_path = r"train_learnable\v2_another_0303\config\train_config.json"
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            # 尝试使用绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "config", "train_config.json")
            if not os.path.exists(config_path):
                # 尝试在当前目录查找
                config_path = "train_config.json"
                if not os.path.exists(config_path):
                    # 列出可能的配置文件
                    logger.error("找不到配置文件，请检查以下位置:")
                    for dirpath, dirnames, filenames in os.walk(current_dir):
                        for filename in filenames:
                            if 'config' in filename and filename.endswith('.json'):
                                logger.info(f"  {os.path.join(dirpath, filename)}")
                    raise FileNotFoundError(f"找不到配置文件")
        
        logger.info(f"📋 使用配置文件: {config_path}")
        
        trainer = AngleTrainerWrapper(config_path)
        trainer.train()
    except KeyboardInterrupt:
        logger.info("⏹️ 训练被用户中断")
    except Exception as e:
        logger.error(f"💥 程序执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()