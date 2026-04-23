import torch
import os
import sys
import gc
import json
import logging
import random
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple

import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments,
    PreTrainedTokenizerBase
)
from transformers.tokenization_utils_base import PaddingStrategy

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================= 工具函数 =======================
def set_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_pooling(outputs, inputs, pooling_strategy, padding_side):
    if pooling_strategy == 'cls':
        outputs = outputs[:, 0]
    elif pooling_strategy == 'cls_avg':
        avg = torch.sum(outputs * inputs["attention_mask"][:, :, None], dim=1) / inputs["attention_mask"].sum(dim=1).unsqueeze(1)
        outputs = (outputs[:, 0] + avg) / 2.0
    elif pooling_strategy == 'last':
        batch_size = inputs['input_ids'].shape[0]
        sequence_lengths = -1 if padding_side == 'left' else inputs["attention_mask"].sum(dim=1) - 1
        outputs = outputs[torch.arange(batch_size, device=outputs.device), sequence_lengths]
    elif pooling_strategy in ['avg', 'mean']:
        outputs = torch.sum(outputs * inputs["attention_mask"][:, :, None], dim=1) / inputs["attention_mask"].sum(dim=1).unsqueeze(1)
    elif pooling_strategy == 'max':
        outputs, _ = torch.max(outputs * inputs["attention_mask"][:, :, None], dim=1)
    else:
        raise NotImplementedError(f'Unsupported pooling: {pooling_strategy}')
    return outputs

# ======================= 复数投影层（与原代码相同） =======================
class ComplexProjection(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.half_size = hidden_size // 2
        self.proj_re = nn.Linear(self.half_size, self.half_size)
        self.proj_im = nn.Linear(self.half_size, self.half_size)
        nn.init.zeros_(self.proj_re.weight)
        nn.init.zeros_(self.proj_im.weight)
        nn.init.zeros_(self.proj_re.bias)
        nn.init.zeros_(self.proj_im.bias)
    def forward(self, x):
        re_orig, im_orig = torch.chunk(x, 2, dim=1)
        re = re_orig + self.proj_re(re_orig)
        im = im_orig + self.proj_im(im_orig)
        return torch.cat([re, im], dim=1)

# ======================= 损失函数（仅保留需要的） =======================
def contrastive_with_negative_loss(text, pos, neg=None, tau=20.0):
    target = torch.cat((pos, neg), dim=0) if neg is not None else pos
    q_norm = F.normalize(text, p=2, dim=1)
    t_norm = F.normalize(target, p=2, dim=1)
    scores = torch.mm(q_norm, t_norm.transpose(0, 1)) * tau
    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
    return nn.CrossEntropyLoss()(scores, labels)

class AngleLoss:
    def __init__(self, ibn_w=1.0, cln_w=1.0, angle_w=0.0, cosine_w=0.0,
                 ibn_tau=20.0, cln_tau=20.0, **kwargs):
        self.ibn_w = ibn_w
        self.cln_w = cln_w
        self.angle_w = angle_w
        self.cosine_w = cosine_w
        self.ibn_tau = ibn_tau
        self.cln_tau = cln_tau
    def __call__(self, labels, outputs):
        # 格式 C: query, positive, negative
        query = outputs[::3]
        positive = outputs[1::3]
        negative = outputs[2::3]
        loss = 0.0
        if self.ibn_w > 0:
            loss += self.ibn_w * contrastive_with_negative_loss(query, positive, tau=self.ibn_tau)
        if self.cln_w > 0:
            loss += self.cln_w * contrastive_with_negative_loss(query, positive, negative, tau=self.cln_tau)
        return loss

# ======================= Pooler =======================
class Pooler:
    def __init__(self, model, pooling_strategy, padding_side):
        self.model = model
        self.pooling_strategy = pooling_strategy
        self.padding_side = padding_side
    def __call__(self, inputs, **kwargs):
        ret = self.model(output_hidden_states=True, **inputs)
        outputs = ret.last_hidden_state
        outputs = get_pooling(outputs, inputs, self.pooling_strategy, self.padding_side)
        return outputs

# ======================= 数据整理器（支持格式C） =======================
def detect_dataset_format(ds):
    if len(ds) > 0:
        sample = ds[0]
        if 'query' in sample and 'positive' in sample and 'negative' in sample:
            return 'C'
    return 'C'

@dataclass
class AngleDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 192
    padding: Union[bool, str] = 'longest'
    def __call__(self, features):
        texts = []
        for f in features:
            texts.extend([f['query'], f['positive'], f['negative']])
        batch = self.tokenizer(texts, padding=self.padding, max_length=self.max_length,
                               truncation=True, return_tensors='pt')
        batch['labels'] = torch.zeros(len(texts), dtype=torch.long)  # dummy
        return batch

# ======================= AngleTrainer（简化版） =======================
class AngleTrainer(Trainer):
    def __init__(self, pooler, loss_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.pooler = pooler
        self.loss_fct = AngleLoss(**loss_kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)
        outputs = self.pooler(inputs)
        actual_model = model.module if hasattr(model, 'module') else model
        if hasattr(actual_model, 'complex_proj'):
            outputs = actual_model.complex_proj(outputs)
        loss = self.loss_fct(labels, outputs)
        return (loss, outputs) if return_outputs else loss

# ======================= AnglE 主类 =======================
class AnglE:
    def __init__(self, model_name_or_path, max_length=192, pooling_strategy='cls', train_mode=True):
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.device = set_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        hidden_size = self.model.config.hidden_size
        self.model.add_module('complex_proj', ComplexProjection(hidden_size))
        if train_mode:
            self.model.train()
        self.pooler = Pooler(self.model, pooling_strategy, self.tokenizer.padding_side)
    def cuda(self):
        self.model = self.model.to(self.device)
        return self
    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
        torch.save(self.model.complex_proj.state_dict(), os.path.join(output_dir, 'complex_proj.bin'))
    @staticmethod
    def from_pretrained(model_name_or_path, max_length=192, pooling_strategy='cls', train_mode=True, **kwargs):
        return AnglE(model_name_or_path, max_length, pooling_strategy, train_mode)
    def fit(self, train_ds, output_dir, batch_size=32, gradient_accumulation_steps=4,
            epochs=5, learning_rate=2e-5, save_steps=1000, warmup_steps=500,
            loss_kwargs=None, logging_steps=100, fp16=False, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        dataset_format = detect_dataset_format(train_ds)
        trainer = AngleTrainer(
            pooler=self.pooler,
            model=self.model,
            loss_kwargs=loss_kwargs or {},
            args=TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                logging_steps=logging_steps,
                save_steps=save_steps,
                fp16=fp16 and torch.cuda.is_available(),
                remove_unused_columns=False,
            ),
            train_dataset=train_ds,
            data_collator=AngleDataCollator(self.tokenizer, max_length=self.max_length),
        )
        trainer.train()
        self.save_pretrained(output_dir)

# ======================= 生成简单三元组数据集 =======================
def generate_simple_triplets(csv_file, max_length=192, tokenizer=None):
    polarity_map = {-2:0, -1:0, 0:1, 1:2, 2:2}
    inv_polarity = {0:"负", 1:"中", 2:"正"}
    reviews_by_key = {}  # (aspect, polarity) -> list of reviews
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        aspect_cols = [col for col in reader.fieldnames if col not in ['id', 'review', 'star']]
        for row in reader:
            review = row['review'].strip()
            if not review:
                continue
            for col in aspect_cols:
                val_str = row.get(col)
                if not val_str:
                    continue
                try:
                    val = int(val_str)
                except:
                    continue
                if val < -2 or val > 2:
                    continue
                polarity = polarity_map[val]
                aspect = col.replace('#', ' ')
                key = (aspect, polarity)
                reviews_by_key.setdefault(key, []).append(review)
    triplets = []
    keys = list(reviews_by_key.keys())
    for (aspect, polarity), pos_reviews in reviews_by_key.items():
        if len(pos_reviews) == 0:
            continue
        positive = random.choice(pos_reviews)
        neg_candidates = []
        for (a, p), revs in reviews_by_key.items():
            if (a == aspect and p == polarity):
                continue
            neg_candidates.extend(revs)
        if not neg_candidates:
            continue
        negative = random.choice(neg_candidates)
        query = f"{aspect} {inv_polarity[polarity]}"
        triplets.append({"query": query, "positive": positive, "negative": negative})
    # 可选：长度过滤
    if tokenizer is not None:
        filtered = []
        for t in triplets:
            q_len = len(tokenizer.encode(t['query'], add_special_tokens=False))
            p_len = len(tokenizer.encode(t['positive'], add_special_tokens=False))
            n_len = len(tokenizer.encode(t['negative'], add_special_tokens=False))
            if q_len <= max_length and p_len <= max_length and n_len <= max_length:
                filtered.append(t)
        logger.info(f"长度过滤：{len(triplets)} -> {len(filtered)}")
        triplets = filtered
    logger.info(f"生成 {len(triplets)} 个简单三元组")
    return triplets

# ======================= 训练包装类 =======================
class SimpleTripletTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    def train(self):
        logger.info("🚀 开始 w/o Hybrid Triplets 消融实验训练")
        # 加载配置
        csv_file = self.config['data']['original_csv_file']
        output_dir = self.config['data']['output_dir']
        model_name = self.config['model']['name']
        max_length = self.config['model']['max_length']
        batch_size = self.config['training']['batch_size']
        grad_acc = self.config['training']['gradient_accumulation_steps']
        epochs = self.config['training']['num_epochs']
        lr = self.config['training']['learning_rate']
        fp16 = self.config['training']['fp16']
        loss_kwargs = self.config['loss']
        # 生成数据
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        train_ds = generate_simple_triplets(csv_file, max_length, tokenizer)
        # 初始化模型
        angle = AnglE.from_pretrained(model_name, max_length=max_length, pooling_strategy='cls', train_mode=True)
        if torch.cuda.is_available():
            angle = angle.cuda()
        # 训练
        angle.fit(
            train_ds=train_ds,
            output_dir=output_dir,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_acc,
            epochs=epochs,
            learning_rate=lr,
            loss_kwargs=loss_kwargs,
            fp16=fp16,
        )
        logger.info(f"训练完成，模型保存至 {output_dir}")

if __name__ == "__main__":
    config_path = r"Ablation_eperience\train\hybrid_triplets\config\train_config.json"
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
    trainer = SimpleTripletTrainer(config_path)
    trainer.train()