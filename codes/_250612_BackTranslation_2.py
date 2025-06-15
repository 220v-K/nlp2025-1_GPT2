# main_integrated.py
import argparse
import csv
import datetime
import math
import os
import random
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (GPT2Tokenizer, MarianMTModel, MarianTokenizer,
                          get_linear_schedule_with_warmup)

# ---------------------------------------------------------------------------
# 0. 유틸리티 함수
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 11711):
    """재현성을 위해 각종 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =================================================================================
# 1. 데이터 증강 (Back-Translation) 모듈
# =================================================================================

class BackTranslator:
    """
    Hugging Face MarianMT 모델을 사용하여 문장을 특정 언어로 번역 후 다시 영어로 역번역합니다.
    """
    def __init__(self, target_lang='es', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.target_lang = target_lang
        
        model_en_to_target = f'Helsinki-NLP/opus-mt-en-{target_lang}'
        model_target_to_en = f'Helsinki-NLP/opus-mt-{target_lang}-en'
        
        print(f"Loading model for en -> {target_lang}...")
        self.tokenizer_en_to_target = MarianTokenizer.from_pretrained(model_en_to_target)
        self.model_en_to_target = MarianMTModel.from_pretrained(model_en_to_target).to(self.device)
        
        print(f"Loading model for {target_lang} -> en...")
        self.tokenizer_target_to_en = MarianTokenizer.from_pretrained(model_target_to_en)
        self.model_target_to_en = MarianMTModel.from_pretrained(model_target_to_en).to(self.device)
        print(f"Models for '{target_lang}' loaded to {self.device}.")

    def translate(self, sentences, batch_size=16):
        """
        문장 리스트를 배치 단위로 역번역합니다.
        """
        augmented_sentences = []
        for i in tqdm(range(0, len(sentences), batch_size), desc=f"Back-translating ({self.target_lang})"):
            batch = sentences[i:i+batch_size]
            
            inputs = self.tokenizer_en_to_target(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            translated_ids = self.model_en_to_target.generate(**inputs)
            translated_texts = self.tokenizer_en_to_target.batch_decode(translated_ids, skip_special_tokens=True)
            
            inputs_back = self.tokenizer_target_to_en(translated_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            back_translated_ids = self.model_target_to_en.generate(**inputs_back)
            back_translated_texts = self.tokenizer_target_to_en.batch_decode(back_translated_ids, skip_special_tokens=True)
            
            augmented_sentences.extend(back_translated_texts)
            
        return augmented_sentences

def run_augmentation(args):
    """
    CSV 파일을 읽어 패러프레이즈 쌍을 역번역으로 증강하고 새 파일에 저장합니다.
    """
    print("="*60)
    print("STEP 1: Starting Data Augmentation using Back-Translation")
    print("="*60)
    print(f"Loading original data from: {args.para_train}")
    df = pd.read_csv(args.para_train, sep='\t')
    
    df_positive = df[df['is_duplicate'] == 1].copy()
    print(f"Found {len(df_positive)} positive pairs to augment.")

    aggregated_rows = []  # 통합 파일 작성을 위한 리스트

    for lang in args.bt_languages:
        translator = BackTranslator(target_lang=lang)

        original_sentences = df_positive['sentence2'].tolist()
        augmented_sentences = translator.translate(original_sentences, batch_size=args.bt_batch_size)

        lang_rows = []
        for i, row in tqdm(
            df_positive.iterrows(), total=len(df_positive), desc=f"Filtering ({lang})"
        ):
            original_s2 = row['sentence2']
            augmented_s2 = augmented_sentences[df_positive.index.get_loc(i)]

            if original_s2.lower().strip() == augmented_s2.lower().strip():
                continue  # 변화가 없으면 스킵

            new_row = {
                'id': str(row['id']) + f'_bt_{lang}',
                'qid1': row['qid1'],
                'qid2': row['qid2'],
                'sentence1': row['sentence1'],
                'sentence2': augmented_s2,
                'is_duplicate': 1,
            }
            lang_rows.append(new_row)
            aggregated_rows.append(new_row)

        # -------------------------------
        # (A) 언어별 파일 저장
        # -------------------------------
        lang_out = Path(args.augmented_train_path).with_name(
            f"{Path(args.augmented_train_path).stem}_{lang}{Path(args.augmented_train_path).suffix}"
        )
        pd.concat([df, pd.DataFrame(lang_rows)], ignore_index=True).to_csv(
            lang_out,
            sep='\t',
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )

        print(
            f"Successfully generated {len(lang_rows):,} augmented samples for language '{lang}'. Saved to {lang_out.name}"
        )

        del translator
        torch.cuda.empty_cache()

    # -------------------------------
    # (B) 통합(aggregated) 파일 저장
    # -------------------------------
    df_augmented = pd.DataFrame(aggregated_rows)
    df_final = (
        pd.concat([df, df_augmented], ignore_index=True)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    print(f"\nOriginal data size: {len(df):,}")
    print(f"Total augmented data size: {len(df_augmented):,}")
    print(f"Final data size: {len(df_final):,}")

    df_final.to_csv(
        args.augmented_train_path,
        sep='\t',
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    print(f"Aggregated data saved to {args.augmented_train_path}")
    print("=" * 60)
    print("Data Augmentation Finished.")
    print("=" * 60)


# =================================================================================
# 2. 모델 및 학습 파이프라인 모듈 (이전과 동일)
# =================================================================================

class GPT2Config:
    def __init__(self, vocab_size=50257, max_position_embeddings=1024, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu_new', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, initializer_range=0.02, layer_norm_eps=1e-5, pad_token_id=50256, **kwargs):
        self.vocab_size, self.max_position_embeddings, self.hidden_size, self.num_hidden_layers, self.num_attention_heads, self.intermediate_size, self.hidden_act, self.hidden_dropout_prob, self.attention_probs_dropout_prob, self.initializer_range, self.layer_norm_eps, self.pad_token_id = vocab_size, max_position_embeddings, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, initializer_range, layer_norm_eps, pad_token_id

def get_extended_attention_mask(attention_mask, dtype):
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

class GPTPreTrainedModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(); self.config = config
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)): module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm): module.bias.data.zero_(); module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None: module.bias.data.zero_()

class GPT2SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads, self.attention_head_size = config.num_attention_heads, int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query, self.key, self.value = nn.Linear(config.hidden_size, self.all_head_size), nn.Linear(config.hidden_size, self.all_head_size), nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    def transpose_for_scores(self, x): return x.view(x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)).permute(0, 2, 1, 3)
    def forward(self, hidden_states, attention_mask):
        query_layer, key_layer, value_layer = self.transpose_for_scores(self.query(hidden_states)), self.transpose_for_scores(self.key(hidden_states)), self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size) + attention_mask
        context_layer = torch.matmul(self.dropout(nn.functional.softmax(attention_scores, dim=-1)), value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        return context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))

class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention, self.attention_dense, self.attention_dropout, self.attention_layer_norm = GPT2SelfAttention(config), nn.Linear(config.hidden_size, config.hidden_size), nn.Dropout(config.hidden_dropout_prob), nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.interm_dense, self.interm_af = nn.Linear(config.hidden_size, config.intermediate_size), nn.GELU() if isinstance(config.hidden_act, str) else config.hidden_act
        self.out_dense, self.out_dropout, self.out_layer_norm = nn.Linear(config.intermediate_size, config.hidden_size), nn.Dropout(config.hidden_dropout_prob), nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention_layer_norm(hidden_states + self.attention_dropout(self.attention_dense(self.self_attention(hidden_states, attention_mask))))
        return self.out_layer_norm(attention_output + self.out_dropout(self.out_dense(self.interm_af(self.interm_dense(attention_output)))))

class GPT2Model(GPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config); self.config = config
        self.word_embedding, self.pos_embedding, self.embed_dropout = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id), nn.Embedding(config.max_position_embeddings, config.hidden_size), nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).unsqueeze(0))
        self.gpt_layers, self.final_layer_norm = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)]), nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.init_weights(); self.dtype = torch.float32
    def embed(self, input_ids): return self.embed_dropout(self.word_embedding(input_ids) + self.pos_embedding(self.position_ids[:, :input_ids.size(1)]))
    def encode(self, hidden_states, attention_mask):
        for layer_module in self.gpt_layers: hidden_states = layer_module(hidden_states, get_extended_attention_mask(attention_mask, self.dtype))
        return hidden_states
    def forward(self, input_ids, attention_mask):
        self.dtype = next(self.parameters()).dtype
        sequence_output = self.final_layer_norm(self.encode(self.embed(input_ids=input_ids), attention_mask=attention_mask))
        return {'last_hidden_state': sequence_output, 'last_token': sequence_output[torch.arange(sequence_output.shape[0]), attention_mask.sum(dim=1) - 1]}
    @classmethod
    def from_pretrained(cls, model_name='gpt2', d=768, l=12, num_heads=12):
        config, our_model, gpt_model = GPT2Config(hidden_size=d, num_hidden_layers=l, num_attention_heads=num_heads), cls(GPT2Config(hidden_size=d, num_hidden_layers=l, num_attention_heads=num_heads)).eval(), MarianMTModel.from_pretrained(model_name).eval() if "Helsinki-NLP" in model_name else GPT2Model.from_pretrained(model_name).eval()
        our_model.word_embedding.load_state_dict(gpt_model.wte.state_dict()); our_model.pos_embedding.load_state_dict(gpt_model.wpe.state_dict())
        for i in range(l):
            layer, gpt_layer = our_model.gpt_layers[i], gpt_model.h[i]
            q_w, k_w, v_w = torch.split(gpt_layer.attn.c_attn.weight.T, d, dim=0); layer.self_attention.query.weight.data, layer.self_attention.key.weight.data, layer.self_attention.value.weight.data = q_w.T, k_w.T, v_w.T
            q_b, k_b, v_b = torch.split(gpt_layer.attn.c_attn.bias, d, dim=0); layer.self_attention.query.bias.data, layer.self_attention.key.bias.data, layer.self_attention.value.bias.data = q_b, k_b, v_b
            layer.attention_dense.load_state_dict(gpt_layer.attn.c_proj.state_dict()); layer.attention_layer_norm.load_state_dict(gpt_layer.ln_1.state_dict())
            layer.interm_dense.weight.data, layer.interm_dense.bias.data = gpt_layer.mlp.c_fc.weight.T, gpt_layer.mlp.c_fc.bias
            layer.out_dense.weight.data, layer.out_dense.bias.data = gpt_layer.mlp.c_proj.weight.T, gpt_layer.mlp.c_proj.bias
            layer.out_layer_norm.load_state_dict(gpt_layer.ln_2.state_dict())
        our_model.final_layer_norm.load_state_dict(gpt_model.ln_f.state_dict())
        return our_model

def preprocess_string(s): return ' '.join(str(s).lower().replace('.', ' .').replace('?', ' ?').replace(',', ' ,').replace('\'', ' \'').split())

def load_paraphrase_data(paraphrase_filename, split='train'):
    paraphrase_data = []
    with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp, delimiter='\t')
        for record in reader:
            try:
                sent_id, s1, s2 = record['id'].lower().strip(), preprocess_string(record['sentence1']), preprocess_string(record['sentence2'])
                if split == 'test': paraphrase_data.append((s1, s2, sent_id))
                else: paraphrase_data.append((s1, s2, int(float(record['is_duplicate'])), sent_id))
            except (KeyError, ValueError): continue
    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
    return paraphrase_data

class ParaphraseDetectionDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset, self.p, self.tokenizer = dataset, args, GPT2Tokenizer.from_pretrained(args.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]
    def collate_fn(self, all_data):
        sent1, sent2, labels, sids = [x[0] for x in all_data], [x[1] for x in all_data], torch.LongTensor([x[2] for x in all_data]), [x[3] for x in all_data]
        cloze = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for s1, s2 in zip(sent1, sent2)]
        enc = self.tokenizer(cloze, return_tensors='pt', padding=True, truncation=True, max_length=self.p.max_length)
        return {'token_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'labels': labels, 'sent_ids': sids}

class ParaphraseDetectionTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset, self.p, self.tokenizer = dataset, args, GPT2Tokenizer.from_pretrained(args.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]
    def collate_fn(self, all_data):
        sent1, sent2, sids = [x[0] for x in all_data], [x[1] for x in all_data], [x[2] for x in all_data]
        cloze = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for s1, s2 in zip(sent1, sent2)]
        enc = self.tokenizer(cloze, return_tensors='pt', padding=True, truncation=True, max_length=self.p.max_length)
        return {'token_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'sent_ids': sids}

class TextFileLogger(pl.Callback):
    def __init__(self, path: Union[str, Path], step_interval: int = 50):
        super().__init__(); self.file, self.step_interval = Path(path), step_interval; self.file.parent.mkdir(parents=True, exist_ok=True); self.dev_results_logged = False
    def _write(self, msg: str):
        with self.file.open("a", encoding="utf-8") as f: f.write(msg + "\n")
    def on_fit_start(self, trainer, pl_module): self._write(f"[START] {datetime.datetime.now().isoformat()}\n  - Run Name: {trainer.logger.name}\n  - Trainable Parameters: {sum(p.numel() for p in pl_module.parameters() if p.requires_grad):,}\n  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n  - Hyperparameters: {pl_module.hparams}")
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.step_interval == 0: self._write(f"[{datetime.datetime.now().isoformat()}] step={trainer.global_step + 1} | loss={outputs['loss'].item():.4f} | lr={trainer.optimizers[0].param_groups[0]['lr']:.3e} | peak_mem={torch.cuda.max_memory_allocated() / (1 << 20) if torch.cuda.is_available() else 0:.0f}MB")
    def _log_epoch_end(self, stage: str, trainer):
        """에폭 종료 시 메트릭을 한 줄로 요약하여 기록합니다."""
        metrics = {
            k: v.item() for k, v in trainer.callback_metrics.items() if k.startswith(stage)
        }
        metric_str = " | ".join(
            f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items()
        )
        self._write(
            f"[{stage.upper():<5}] EPOCH {trainer.current_epoch} END: {metric_str}"
        )
    def on_train_epoch_end(self, trainer, pl_module): self._log_epoch_end("train", trainer)
    def on_validation_epoch_end(self, trainer, pl_module): self._log_epoch_end("val", trainer)
    def on_test_end(self, trainer, pl_module):
        if self.dev_results_logged: return
        self._write("\n" + "="*50 + "\n[DEV SET EVALUATION RESULTS]\n" + "="*50)
        preds, labels = [], []
        for o in pl_module.test_outputs:
            if o['dataset_idx'] == 0: preds.extend(o['prediction'].cpu().numpy()); labels.extend(o['labels'].cpu().numpy())
        if not labels: self._write("No dev set results found."); return
        self._write(f"  - Accuracy:  {accuracy_score(labels, preds):.4f}\n  - F1-Score:  {f1_score(labels, preds, average='macro'):.4f}\n  - Precision: {precision_score(labels, preds, average='macro', zero_division=0):.4f}\n  - Recall:    {recall_score(labels, preds, average='macro', zero_division=0):.4f}\n" + "="*50 + "\n")
        self.dev_results_logged = True

class ParaphraseDataModuleBET(pl.LightningDataModule):
    def __init__(self, args): super().__init__(); self.save_hyperparameters(args)
    def setup(self, stage=None):
        train_path = self.hparams.augmented_train_path
        if stage in ("fit", None):
            self.train_set = ParaphraseDetectionDataset(load_paraphrase_data(train_path, 'train'), self.hparams)
            self.val_set = ParaphraseDetectionDataset(load_paraphrase_data(self.hparams.para_dev, 'dev'), self.hparams)
        if stage in ("test", None):
            self.dev4test_set = ParaphraseDetectionDataset(load_paraphrase_data(self.hparams.para_dev, 'dev'), self.hparams)
            self.test_set = ParaphraseDetectionTestDataset(load_paraphrase_data(self.hparams.para_test, 'test'), self.hparams)
    def train_dataloader(self): return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, collate_fn=self.train_set.collate_fn, pin_memory=True)
    def val_dataloader(self): return DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, collate_fn=self.val_set.collate_fn, pin_memory=True)
    def test_dataloader(self): return [DataLoader(self.dev4test_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, collate_fn=self.dev4test_set.collate_fn), DataLoader(self.test_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, collate_fn=self.test_set.collate_fn)]

class ParaphraseLitModuleBET(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__(); self.save_hyperparameters(hparams); self._build_model()
        self.train_acc, self.val_acc, self.val_f1 = torchmetrics.Accuracy(task="multiclass", num_classes=2), torchmetrics.Accuracy(task="multiclass", num_classes=2), torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
        self.test_outputs = []
    def _build_model(self):
        d, (l, h) = {'gpt2': 768, 'gpt2-medium': 1024, 'gpt2-large': 1280}[self.hparams.model_size], {'gpt2': (12, 12), 'gpt2-medium': (24, 16), 'gpt2-large': (36, 20)}[self.hparams.model_size]
        self.gpt, self.classification_head = GPT2Model.from_pretrained(self.hparams.model_size, d=d, l=l, num_heads=h), nn.Linear(d, 2)
        for p in self.parameters(): p.requires_grad = True
    def forward(self, input_ids, attention_mask): return self.classification_head(self.gpt(input_ids=input_ids, attention_mask=attention_mask)['last_token'])
    def _compute_loss(self, logits1, logits2, labels):
        ce_loss = (F.cross_entropy(logits1, labels, label_smoothing=self.hparams.label_smoothing) + F.cross_entropy(logits2, labels, label_smoothing=self.hparams.label_smoothing)) / 2
        kl_loss = self.hparams.rdrop_alpha * F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='batchmean')
        return ce_loss + kl_loss, ce_loss, kl_loss
    def training_step(self, batch, batch_idx):
        ids, mask, labels = batch["token_ids"], batch["attention_mask"], batch["labels"].flatten()
        logits1, logits2 = self(ids, mask), self(ids, mask)
        total_loss, ce_loss, kl_loss = self._compute_loss(logits1, logits2, labels)
        self.log_dict({"train/loss": total_loss, "train/ce_loss": ce_loss, "train/kl_loss": kl_loss}, on_step=True, on_epoch=True, prog_bar=True, batch_size=ids.size(0))
        self.train_acc(torch.argmax(logits1, dim=1), labels); self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss
    def validation_step(self, batch, batch_idx):
        ids, mask, labels = batch["token_ids"], batch["attention_mask"], batch["labels"].flatten()
        logits = self(ids, mask)
        preds = torch.argmax(logits, dim=1); self.val_acc.update(preds, labels); self.val_f1.update(preds, labels)
        self.log_dict({"val/loss": F.cross_entropy(logits, labels), "val/acc": self.val_acc, "val/f1": self.val_f1}, on_epoch=True, prog_bar=True, batch_size=ids.size(0))
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output = {"sent_ids": batch["sent_ids"], "prediction": torch.argmax(self(batch["token_ids"], batch["attention_mask"]), dim=1), "dataset_idx": dataloader_idx}
        if "labels" in batch: output["labels"] = batch["labels"].flatten()
        self.test_outputs.append(output)
    def on_test_epoch_end(self):
        output_dir = Path(self.hparams.output_dir) / "predictions"; output_dir.mkdir(parents=True, exist_ok=True)
        dev_path, test_path = output_dir/"para-dev-output.csv", output_dir/"para-test-output.csv"
        with open(dev_path, "w+", newline='', encoding='utf-8') as f_dev, open(test_path, "w+", newline='', encoding='utf-8') as f_test:
            w_dev, w_test = csv.writer(f_dev), csv.writer(f_test); w_dev.writerow(["id", "Predicted_Is_Paraphrase"]); w_test.writerow(["id", "Predicted_Is_Paraphrase"])
            for o in self.test_outputs:
                writer = w_dev if o['dataset_idx'] == 0 else w_test
                for sid, pred in zip(o['sent_ids'], o['prediction'].tolist()): writer.writerow([sid, pred])
        print(f"\nDev predictions saved to {dev_path}\nTest predictions saved to {test_path}")
        self.test_outputs.clear()
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(self.trainer.estimated_stepping_batches * self.hparams.warmup_ratio), num_training_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

# =================================================================================
# 3. 메인 실행 블록
# =================================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Integrated Pipeline for Paraphrase Detection with Back-Translation")
    
    # --- 파일 경로 ---
    parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs")
    parser.add_argument("--augmented_train_path", type=str, default=None, help="Path for the augmented data file. If None, it will be auto-generated.")

    # --- 데이터 증강 옵션 ---
    parser.add_argument("--force_augment", action='store_true', help="Force re-augmentation even if the augmented file exists.")
    parser.add_argument("--bt_languages", nargs='+', default=['es', 'de', 'fr', 'ko', 'ja', 'zh'], help="Intermediate languages for back-translation.")
    parser.add_argument("--bt_batch_size", type=int, default=512, help="Batch size for back-translation process.")

    # --- 모델 및 학습 하이퍼파라미터 ---
    parser.add_argument("--model_size", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--rdrop_alpha", type=float, default=4.0)
    
    # --- 시스템 ---
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--gpus", type=int, default=1)
    
    args = parser.parse_args()
    
    # 증강 파일 경로 자동 설정
    if args.augmented_train_path is None:
        p = Path(args.para_train)
        args.augmented_train_path = p.parent / f"{p.stem}-augmented-bt.csv"

    return args

def main():
    args = get_args()
    seed_everything(args.seed)

    # --- 1단계: 데이터 증강 (필요시 실행) ---
    augmented_file = Path(args.augmented_train_path)
    if args.force_augment or not augmented_file.exists():
        run_augmentation(args)
    else:
        print(f"Found existing augmented file at {augmented_file}, skipping augmentation.")

    # --- 2단계: 모델 학습 ---
    print("\n" + "="*60)
    print("STEP 2: Starting Model Training")
    print("="*60)
    
    output_path = Path(args.output_dir)
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model_size}_BT_Integrated"
    
    datamodule = ParaphraseDataModuleBET(args)
    model = ParaphraseLitModuleBET(hparams=args)
    
    # 로거 및 콜백 설정
    tb_logger = TensorBoardLogger(output_path / "logs", name=run_name)
    txt_logger = TextFileLogger(output_path / "logs" / run_name / "train_log.txt")
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints" / run_name,
        filename="best-{epoch}-{val/acc:.4f}",
        save_top_k=1, monitor="val/acc", mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="gpu" if args.gpus > 0 and torch.cuda.is_available() else "cpu",
        devices=args.gpus, precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor, txt_logger],
        logger=tb_logger,
        log_every_n_steps=50, deterministic="warn"
    )
    
    trainer.fit(model, datamodule)
    
    print("\n--- Starting Testing ---")
    trainer.test(datamodule=datamodule, ckpt_path='best')
    
    print(f"\n--- Pipeline Finished ---\nFind results in: {output_path.resolve()}")

if __name__ == "__main__":
    main()

# 실행 예시:
# python main_integrated.py --para_train data/quora-train.csv --bt_languages es de