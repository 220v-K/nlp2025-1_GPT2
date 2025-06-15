# main.py

import argparse
import csv
import datetime
import os
import random
import re
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

# =================================================================================
# Collaborator Module: models/gpt2.py & etc.
# 이 섹션은 제공된 협력자 모듈의 코드를 포함합니다.
# =================================================================================

# --- config.py의 내용 ---
class GPT2Config:
    def __init__(self,
                 vocab_size=50257,
                 max_position_embeddings=1024,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act='gelu_new',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 layer_norm_eps=1e-5,
                 pad_token_id=50256,
                 **kwargs):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id

# --- utils.py의 내용 ---
def get_extended_attention_mask(attention_mask, dtype):
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

# --- models/base_gpt.py의 내용 ---
class GPTPreTrainedModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

# --- modules/gpt2_layer.py의 내용 ---
class GPT2SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = GPT2SelfAttention(config)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.interm_af = nn.GELU()
        else:
            self.interm_af = config.hidden_act
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.self_attention(hidden_states, attention_mask)
        attention_output = self.attention_dense(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(hidden_states + attention_output)
        intermediate_output = self.interm_dense(attention_output)
        intermediate_output = self.interm_af(intermediate_output)
        layer_output = self.out_dense(intermediate_output)
        layer_output = self.out_dropout(layer_output)
        layer_output = self.out_layer_norm(attention_output + layer_output)
        return layer_output

# --- models/gpt2.py의 내용 ---
class GPT2Model(GPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)
        self.gpt_layers = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.init_weights()
        self.dtype = torch.float32

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        inputs_embeds = self.word_embedding(input_ids)
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)
        embedding_output = inputs_embeds + pos_embeds
        embedding_output = self.embed_dropout(embedding_output)
        return embedding_output

    def encode(self, hidden_states, attention_mask):
        extended_attention_mask = get_extended_attention_mask(attention_mask, self.dtype)
        for i, layer_module in enumerate(self.gpt_layers):
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states

    def forward(self, input_ids, attention_mask):
        self.dtype = self.query.weight.dtype if hasattr(self, 'query') else torch.float32 # For mixed precision
        embedding_output = self.embed(input_ids=input_ids)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        sequence_output = self.final_layer_norm(sequence_output)
        last_non_pad_idx = attention_mask.sum(dim=1) - 1
        last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]
        return {'last_hidden_state': sequence_output, 'last_token': last_token}

    @classmethod
    def from_pretrained(cls, model_name='gpt2', d=768, l=12, num_heads=12):
        from transformers import GPT2Model as OpenAIGPT2Model
        config = GPT2Config(hidden_size=d, num_hidden_layers=l, num_attention_heads=num_heads)
        our_model = cls(config).eval()
        gpt_model = OpenAIGPT2Model.from_pretrained(model_name).eval()

        our_model.word_embedding.load_state_dict(gpt_model.wte.state_dict())
        our_model.pos_embedding.load_state_dict(gpt_model.wpe.state_dict())

        for i in range(l):
            layer = our_model.gpt_layers[i]
            gpt_layer = gpt_model.h[i]
            
            # Attention
            qkv_weight = gpt_layer.attn.c_attn.weight.T
            q_w, k_w, v_w = torch.split(qkv_weight, d, dim=0)
            layer.self_attention.query.weight.data = q_w.T
            layer.self_attention.key.weight.data = k_w.T
            layer.self_attention.value.weight.data = v_w.T
            
            qkv_bias = gpt_layer.attn.c_attn.bias
            q_b, k_b, v_b = torch.split(qkv_bias, d, dim=0)
            layer.self_attention.query.bias.data = q_b
            layer.self_attention.key.bias.data = k_b
            layer.self_attention.value.bias.data = v_b
            
            layer.attention_dense.load_state_dict(gpt_layer.attn.c_proj.state_dict())
            layer.attention_layer_norm.load_state_dict(gpt_layer.ln_1.state_dict())
            
            # MLP (Conv1D -> Linear). HuggingFace Conv1D stores weight as (in_features, out_features),
            # whereas nn.Linear expects (out_features, in_features). Transpose is therefore required.

            layer.interm_dense.weight.data = gpt_layer.mlp.c_fc.weight.T  # shape: (3072, 768)
            layer.interm_dense.bias.data   = gpt_layer.mlp.c_fc.bias

            layer.out_dense.weight.data   = gpt_layer.mlp.c_proj.weight.T  # shape: (768, 3072)
            layer.out_dense.bias.data     = gpt_layer.mlp.c_proj.bias

        our_model.final_layer_norm.load_state_dict(gpt_model.ln_f.state_dict())
        return our_model

# =================================================================================
# Collaborator Module: datasets.py
# 이 섹션은 논문의 BET 방법론을 적용하여 수정된 데이터셋 코드를 포함합니다.
# =================================================================================

def preprocess_string(s):
    return ' '.join(s.lower().replace('.', ' .').replace('?', ' ?').replace(',', ' ,').replace('\'', ' \'').split())

def load_paraphrase_data(paraphrase_filename, split='train', augment=False):
    """
    데이터를 로드하고 BET 논문에 기반한 데이터 증강을 적용합니다.
    - augment=True 이고 split='train'일 때, is_duplicate=1인 모든 샘플 (s1, s2)에 대해
      대칭적인 샘플 (s2, s1)을 추가하여 패러프레이즈 쌍을 늘립니다.
    """
    paraphrase_data = []
    
    if split == 'test':
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))
    else:
        original_data = []
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    s1 = preprocess_string(record['sentence1'])
                    s2 = preprocess_string(record['sentence2'])
                    label = int(float(record['is_duplicate']))
                    original_data.append((s1, s2, label, sent_id))
                except (KeyError, ValueError):
                    pass
        
        paraphrase_data.extend(original_data)
        
        # BET 방법론 적용: 데이터 증강
        if augment and split == 'train':
            augmented_samples = 0
            for s1, s2, label, sent_id in original_data:
                if label == 1: # 패러프레이즈인 경우에만 증강
                    paraphrase_data.append((s2, s1, 1, sent_id + '_aug'))
                    augmented_samples += 1
            print(f"Augmented {augmented_samples} samples using BET-like strategy (s1 <-> s2).")

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
    return paraphrase_data

class ParaphraseDetectionDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.max_length = getattr(args, 'max_length', 128)
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        labels = torch.LongTensor([x[2] for x in all_data])
        sent_ids = [x[3] for x in all_data]
        
        cloze_style_sents = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for (s1, s2) in zip(sent1, sent2)]
        encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        
        return {
            'token_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels,
            'sent_ids': sent_ids
        }

class ParaphraseDetectionTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.max_length = getattr(args, 'max_length', 128)
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        sent1 = [x[0] for x in all_data]
        sent2 = [x[1] for x in all_data]
        sent_ids = [x[2] for x in all_data]
        
        cloze_style_sents = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for (s1, s2) in zip(sent1, sent2)]
        encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        
        return {
            'token_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'sent_ids': sent_ids
        }

# =================================================================================
# PyTorch Lightning 파이프라인
# =================================================================================

def seed_everything(seed: int = 11711):
    """재현성을 위한 시드 고정 함수."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TextFileLogger(pl.Callback):
    """학습 진행 상황을 텍스트 파일에 기록하는 로거."""
    def __init__(self, path: Union[str, Path], step_interval: int = 50):
        super().__init__()
        self.file = Path(path)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.step_interval = step_interval
        self._start_time = None
        self.dev_results_logged = False

    def _write(self, msg: str):
        with self.file.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def on_fit_start(self, trainer, pl_module):
        self._start_time = datetime.datetime.now()
        n_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self._write(f"[START] {self._start_time.isoformat()}")
        self._write(f"  - Run Name: {trainer.logger.name}")
        self._write(f"  - Trainable Parameters: {n_params:,}")
        self._write(f"  - Device: {gpu_name}")
        self._write(f"  - Hyperparameters: {pl_module.hparams}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.step_interval != 0:
            return
        loss_val = outputs["loss"].item()
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        mem = torch.cuda.max_memory_allocated() / (1 << 20) if torch.cuda.is_available() else 0
        self._write(f"[{datetime.datetime.now().isoformat()}] step={trainer.global_step + 1} | "
                    f"loss={loss_val:.4f} | lr={lr:.3e} | peak_mem={mem:.0f}MB")

    def _log_epoch_end(self, stage: str, trainer):
        metrics = {k: v.item() for k, v in trainer.callback_metrics.items() if stage in k}
        metric_str = " | ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items())
        self._write(f"[{stage.upper():<5}] EPOCH {trainer.current_epoch} END: {metric_str}")

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_epoch_end("train", trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_epoch_end("val", trainer)

    def on_test_end(self, trainer, pl_module):
        if self.dev_results_logged: return
        
        self._write("\n" + "="*50)
        self._write(f"[DEV SET EVALUATION RESULTS]")
        self._write("="*50)

        all_preds = []
        all_labels = []

        # pl_module.test_outputs는 test_step에서 채워짐
        for output in pl_module.test_outputs:
            if output['dataset_idx'] == 0: # 0은 dev set을 의미
                all_preds.extend(output['prediction'].cpu().numpy())
                all_labels.extend(output['labels'].cpu().numpy())
        
        if not all_labels:
            self._write("No dev set results found to log.")
            return

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')

        self._write(f"  - Accuracy:  {acc:.4f}")
        self._write(f"  - F1-Score:  {f1:.4f}")
        self._write(f"  - Precision: {precision:.4f}")
        self._write(f"  - Recall:    {recall:.4f}")
        self._write("="*50 + "\n")
        self.dev_results_logged = True


class ParaphraseDataModuleBET(pl.LightningDataModule):
    """BET 증강을 사용하는 데이터 모듈."""
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

    def setup(self, stage=None):
        if stage in ("fit", None):
            train_raw = load_paraphrase_data(self.hparams.para_train, split='train', augment=True)
            dev_raw = load_paraphrase_data(self.hparams.para_dev, split='dev', augment=False)
            self.train_set = ParaphraseDetectionDataset(train_raw, self.hparams)
            self.val_set = ParaphraseDetectionDataset(dev_raw, self.hparams)

        if stage in ("test", None):
            dev_raw = load_paraphrase_data(self.hparams.para_dev, split='dev', augment=False)
            test_raw = load_paraphrase_data(self.hparams.para_test, split='test', augment=False)
            self.dev4test_set = ParaphraseDetectionDataset(dev_raw, self.hparams) # 레이블 포함
            self.test_set = ParaphraseDetectionTestDataset(test_raw, self.hparams) # 레이블 미포함

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, 
                          num_workers=self.hparams.num_workers, collate_fn=self.train_set.collate_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, collate_fn=self.val_set.collate_fn, pin_memory=True)

    def test_dataloader(self):
        return [
            DataLoader(self.dev4test_set, batch_size=self.hparams.batch_size, shuffle=False,
                       num_workers=self.hparams.num_workers, collate_fn=self.dev4test_set.collate_fn, pin_memory=True),
            DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False,
                       num_workers=self.hparams.num_workers, collate_fn=self.test_set.collate_fn, pin_memory=True)
        ]


class ParaphraseLitModuleBET(pl.LightningModule):
    """BET 데이터 증강과 R-Drop 정규화를 사용한 GPT-2 패러프레이즈 탐지 모델."""
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self._build_model()

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
        
        # Test output buffer
        self.test_outputs = []

    def _build_model(self):
        model_config_map = {'gpt2': 768, 'gpt2-medium': 1024, 'gpt2-large': 1280}
        d = model_config_map[self.hparams.model_size]
        l, num_heads = {'gpt2': (12, 12), 'gpt2-medium': (24, 16), 'gpt2-large': (36, 20)}[self.hparams.model_size]

        self.gpt = GPT2Model.from_pretrained(self.hparams.model_size, d=d, l=l, num_heads=num_heads)
        self.classification_head = nn.Linear(d, 2)
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_token_hidden_state = gpt_out['last_token']
        logits = self.classification_head(last_token_hidden_state)
        return logits

    def _compute_loss(self, logits1, logits2, labels):
        ce_loss_1 = F.cross_entropy(logits1, labels, label_smoothing=self.hparams.label_smoothing)
        ce_loss_2 = F.cross_entropy(logits2, labels, label_smoothing=self.hparams.label_smoothing)
        ce_loss = (ce_loss_1 + ce_loss_2) / 2
        
        kl_div = F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='batchmean')
        kl_loss = self.hparams.rdrop_alpha * kl_div
        
        total_loss = ce_loss + kl_loss
        return total_loss, ce_loss, kl_loss

    def training_step(self, batch, batch_idx):
        ids, mask, labels = batch["token_ids"], batch["attention_mask"], batch["labels"].flatten()
        
        logits1 = self(ids, mask)
        logits2 = self(ids, mask)
        
        total_loss, ce_loss, kl_loss = self._compute_loss(logits1, logits2, labels)
        
        self.log_dict({
            "train/loss": total_loss, "train/ce_loss": ce_loss, "train/kl_loss": kl_loss
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=ids.size(0))

        preds = torch.argmax(logits1, dim=1)
        self.train_acc(preds, labels)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        ids, mask, labels = batch["token_ids"], batch["attention_mask"], batch["labels"].flatten()
        logits = self(ids, mask)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)
        
        self.log_dict({
            "val/loss": loss, "val/acc": self.val_acc, "val/f1": self.val_f1
        }, on_epoch=True, prog_bar=True, logger=True, batch_size=ids.size(0))
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        ids, mask, sent_ids = batch["token_ids"], batch["attention_mask"], batch["sent_ids"]
        logits = self(ids, mask)
        preds = torch.argmax(logits, dim=1)

        output = {"sent_ids": sent_ids, "prediction": preds, "dataset_idx": dataloader_idx}
        if "labels" in batch:
            output["labels"] = batch["labels"].flatten()
        
        self.test_outputs.append(output)

    def on_test_epoch_end(self):
        output_dir = Path(self.hparams.output_dir) / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        dev_out_path = output_dir / "para-dev-output.csv"
        test_out_path = output_dir / "para-test-output.csv"

        with open(dev_out_path, "w+", newline='', encoding='utf-8') as f_dev, \
             open(test_out_path, "w+", newline='', encoding='utf-8') as f_test:
            
            writer_dev = csv.writer(f_dev)
            writer_test = csv.writer(f_test)
            writer_dev.writerow(["id", "Predicted_Is_Paraphrase"])
            writer_test.writerow(["id", "Predicted_Is_Paraphrase"])
            
            for output in self.test_outputs:
                for sid, pred in zip(output['sent_ids'], output['prediction'].tolist()):
                    if output['dataset_idx'] == 0: # dev
                        writer_dev.writerow([sid, pred])
                    else: # test
                        writer_test.writerow([sid, pred])

        print(f"\nDev predictions saved to {dev_out_path}")
        print(f"Test predictions saved to {test_out_path}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        
        # Scheduler
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * self.hparams.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

# =================================================================================
# 메인 실행 스크립트
# =================================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Paraphrase Detection with BET and R-Drop")
    
    # Paths
    parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs")

    # Model and Training Hyperparameters
    parser.add_argument("--model_size", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--rdrop_alpha", type=float, default=4.0, help="Weight for R-Drop KL loss")
    
    # System
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    seed_everything(args.seed)

    output_path = Path(args.output_dir)
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model_size}_BET_RDrop"

    datamodule = ParaphraseDataModuleBET(args)
    model = ParaphraseLitModuleBET(hparams=args)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints" / run_name,
        filename="best-{epoch}-{val/acc:.4f}",
        save_top_k=3,
        monitor="val/acc",
        mode="max",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Loggers
    tb_logger = TensorBoardLogger(output_path / "logs", name=run_name)
    txt_logger = TextFileLogger(output_path / "logs" / run_name / "train_log.txt")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus > 0 and torch.cuda.is_available() else "cpu",
        devices=args.gpus,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor, txt_logger],
        logger=tb_logger,
        log_every_n_steps=50,
        deterministic="warn"
    )
    
    print("--- Starting Training ---")
    trainer.fit(model, datamodule)
    
    print("\n--- Starting Testing ---")
    trainer.test(datamodule=datamodule, ckpt_path='best')
    
    print("\n--- Pipeline Finished ---")
    print(f"Find your results in: {output_path.resolve()}")
    print(f"  - Logs (TensorBoard): {output_path / 'logs' / run_name}")
    print(f"  - Logs (Text): {output_path / 'logs' / run_name / 'train_log.txt'}")
    print(f"  - Checkpoints: {output_path / 'checkpoints' / run_name}")
    print(f"  - Predictions: {output_path / 'predictions'}")

if __name__ == "__main__":
    main()