# main_pl_contrastive.py
import argparse
import csv
import datetime
import os
import random
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2Model as HFGPT2Model
from torch.nn.utils.rnn import pad_sequence

#! 주의: 메모리 문제 발생 시 주석 처리
torch.set_float32_matmul_precision('high')

# =================================================================================
# 1. 설정 및 유틸리티 함수 (변경 없음)
# =================================================================================

def seed_everything(seed: int = 11711):
    """재현성을 위한 시드 고정 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def preprocess_string(s: str) -> str:
    """간단한 텍스트 전처리기"""
    return ' '.join(s.lower().replace('.', ' .').replace('?', ' ?').replace(',', ' ,').replace('\'', ' \'').split())

def setup_logging(log_dir: Path) -> logging.Logger:
    """파일 및 콘솔에 동시에 출력되는 로거를 설정합니다."""
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = log_dir / f'paraphrase_detection_{current_time}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"로깅 시작. 로그 파일: {log_file}")
    return logger

def load_paraphrase_data(paraphrase_filename: str, split: str = 'train'):
    """Quora 데이터셋을 로드합니다."""
    data = []
    if split == 'test':
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp, delimiter='\t')
            for record in reader:
                data.append({
                    'sentence1': preprocess_string(record['sentence1']),
                    'sentence2': preprocess_string(record['sentence2']),
                    'id': record['id'].lower().strip()
                })
    else:
        with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp, delimiter='\t')
            for record in reader:
                try:
                    data.append({
                        'sentence1': preprocess_string(record['sentence1']),
                        'sentence2': preprocess_string(record['sentence2']),
                        'is_duplicate': int(float(record['is_duplicate'])),
                        'id': record['id'].lower().strip()
                    })
                except (KeyError, ValueError):
                    pass
    print(f"Loaded {len(data)} {split} examples from {paraphrase_filename}")
    return data

# =================================================================================
# 2. 데이터셋 클래스 (⭐️ 수정됨: 대조 학습을 위해 문장을 별도로 토크나이징)
# =================================================================================

class ParaphraseDataset(Dataset):
    def __init__(self, dataset: list, tokenizer: GPT2Tokenizer, max_length: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        s1, s2 = item['sentence1'], item['sentence2']
        label, sent_id = item['is_duplicate'], item['id']

        encoding1 = self.tokenizer(s1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        encoding2 = self.tokenizer(s2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            's1_token_ids': encoding1['input_ids'].squeeze(0),
            's1_attention_mask': encoding1['attention_mask'].squeeze(0),
            's2_token_ids': encoding2['input_ids'].squeeze(0),
            's2_attention_mask': encoding2['attention_mask'].squeeze(0),
            'labels': torch.tensor([label], dtype=torch.float),
            'sent_ids': sent_id
        }

# -------------- 추가: 커스텀 collate_fn --------------

def paraphrase_collate_fn(batch: list) -> dict:
    """DataLoader가 배치 텐서를 스택할 때 발생하는 storage resize 오류를 방지하기 위한 커스텀 collate 함수"""
    # ---------------------------------------------------------------------------------
    # 일부 문장이 비어 있거나 토크나이저 이슈로 인해 길이가 달라질 수 있습니다.
    # pad_sequence 를 이용해 가장 긴 시퀀스 길이에 맞추어 padding 하여 배치 텐서를 생성합니다.
    # GPT-2 의 pad_token_id (default 50256) 를 모를 수도 있으므로 0 으로 패딩합니다. 
    # 어텐션 마스크는 패딩 위치에 0, 실제 토큰 위치에 1 이 되도록 합니다.
    # ---------------------------------------------------------------------------------
    s1_token_ids = pad_sequence([item['s1_token_ids'] for item in batch], batch_first=True, padding_value=0)
    s1_attention_mask = pad_sequence([item['s1_attention_mask'] for item in batch], batch_first=True, padding_value=0)
    s2_token_ids = pad_sequence([item['s2_token_ids'] for item in batch], batch_first=True, padding_value=0)
    s2_attention_mask = pad_sequence([item['s2_attention_mask'] for item in batch], batch_first=True, padding_value=0)

    # sent_ids 는 문자열이므로 리스트 유지
    sent_ids = [item['sent_ids'] for item in batch]

    batch_dict = {
        's1_token_ids': s1_token_ids,
        's1_attention_mask': s1_attention_mask,
        's2_token_ids': s2_token_ids,
        's2_attention_mask': s2_attention_mask,
        'sent_ids': sent_ids
    }

    # 학습/검증 배치에는 labels 가 존재함
    if 'labels' in batch[0]:
        labels = torch.tensor([item['labels'].item() for item in batch], dtype=torch.float).unsqueeze(1)
        batch_dict['labels'] = labels

    return batch_dict

class ParaphraseTestDataset(Dataset):
    def __init__(self, dataset: list, tokenizer: GPT2Tokenizer, max_length: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        s1, s2 = item['sentence1'], item['sentence2']
        sent_id = item['id']

        encoding1 = self.tokenizer(s1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        encoding2 = self.tokenizer(s2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            's1_token_ids': encoding1['input_ids'].squeeze(0),
            's1_attention_mask': encoding1['attention_mask'].squeeze(0),
            's2_token_ids': encoding2['input_ids'].squeeze(0),
            's2_attention_mask': encoding2['attention_mask'].squeeze(0),
            'sent_ids': sent_id
        }

# =================================================================================
# 3. PyTorch Lightning 데이터 모듈 (변경 없음)
# =================================================================================

class ParaphraseDataModule(pl.LightningDataModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage: str = None):
        if stage in ("fit", None):
            train_raw = load_paraphrase_data(self.hparams.para_train, split='train')
            dev_raw = load_paraphrase_data(self.hparams.para_dev, split='dev')
            self.train_set = ParaphraseDataset(train_raw, self.tokenizer, self.hparams.max_length)
            self.val_set = ParaphraseDataset(dev_raw, self.tokenizer, self.hparams.max_length)
        if stage in ("test", None):
            dev_raw = load_paraphrase_data(self.hparams.para_dev, split='dev')
            test_raw = load_paraphrase_data(self.hparams.para_test, split='test')
            self.dev_for_test_set = ParaphraseDataset(dev_raw, self.tokenizer, self.hparams.max_length)
            self.test_set = ParaphraseTestDataset(test_raw, self.tokenizer, self.hparams.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=paraphrase_collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=paraphrase_collate_fn)
    def test_dataloader(self):
        return [
            DataLoader(self.dev_for_test_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=paraphrase_collate_fn),
            DataLoader(self.test_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=paraphrase_collate_fn)
        ]

# =================================================================================
# 4. ⭐️ 새로운 기능: 대조 손실 함수
# =================================================================================

class ContrastiveLoss(nn.Module):
    """
    대조 손실 함수. 긍정 쌍의 거리는 좁히고 부정 쌍의 거리는 margin 이상으로 넓힙니다.
    """
    def __init__(self, margin: float = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9 # 부동 소수점 안정성

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, label: torch.Tensor):
        # 유클리드 거리 계산
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)

        # 손실 계산
        # label=1 (긍정): 거리를 제곱하여 손실로 사용 (거리가 0에 가까워지도록)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# =================================================================================
# 5. PyTorch Lightning 모델 (⭐️ 수정됨: 대조 학습 로직으로 전면 재구성)
# =================================================================================

class ParaphraseLitModule(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.gpt = HFGPT2Model.from_pretrained(self.hparams.model_size)
        
        # 대조 학습에서는 별도의 classification head가 필요 없습니다.
        # 모든 파라미터 학습
        for param in self.parameters():
            param.requires_grad = True

        # 손실 함수 정의
        self.criterion = ContrastiveLoss(margin=self.hparams.margin)

        # Metric 정의 (이제 0과 1로 예측하므로 이진 분류 task로 변경)
        self.train_acc = torchmetrics.Accuracy(task="binary", threshold=0.5)
        self.val_acc = torchmetrics.Accuracy(task="binary", threshold=0.5)
        self.val_f1 = torchmetrics.F1Score(task="binary", threshold=0.5)
        self.test_outputs = []

    def _get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """주어진 입력에 대해 문장 임베딩을 추출합니다."""
        gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = gpt_out.last_hidden_state

        pooled_output = None
        if self.hparams.pooling_method == 'last':
            last_idx = attention_mask.sum(dim=1) - 1
            pooled_output = hidden_states[torch.arange(hidden_states.size(0)), last_idx]
        elif self.hparams.pooling_method == 'mean':
            expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden_states = torch.sum(hidden_states * expanded_mask, dim=1)
            sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
            pooled_output = sum_hidden_states / sum_mask
        elif self.hparams.pooling_method == 'cls':
            pooled_output = hidden_states[:, 0, :]
        else:
            raise ValueError("Unsupported pooling method. Choose from 'last', 'mean', 'cls'.")
        
        return pooled_output

    def forward(self, s1_ids: torch.Tensor, s1_mask: torch.Tensor, s2_ids: torch.Tensor, s2_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """두 문장에 대한 임베딩을 각각 반환합니다."""
        embedding1 = self._get_embedding(s1_ids, s1_mask)
        embedding2 = self._get_embedding(s2_ids, s2_mask)
        return embedding1, embedding2

    def training_step(self, batch, batch_idx):
        embedding1, embedding2 = self(batch['s1_token_ids'], batch['s1_attention_mask'], batch['s2_token_ids'], batch['s2_attention_mask'])
        loss = self.criterion(embedding1, embedding2, batch['labels'])
        
        # 정확도 계산을 위한 거리 기반 예측
        distance = F.pairwise_distance(embedding1, embedding2)
        # 거리가 작을수록 유사(1), 클수록 비유사(0). 따라서 1에서 거리를 빼서 유사도 점수로 변환
        # margin의 절반을 임계값으로 사용하도록 정규화
        similarity_score = 1 - (distance / self.hparams.margin)
        self.train_acc(similarity_score, batch['labels'].view(-1).int())

        batch_size = batch['labels'].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        embedding1, embedding2 = self(batch['s1_token_ids'], batch['s1_attention_mask'], batch['s2_token_ids'], batch['s2_attention_mask'])
        loss = self.criterion(embedding1, embedding2, batch['labels'])
        
        distance = F.pairwise_distance(embedding1, embedding2)
        similarity_score = 1 - (distance / self.hparams.margin)
        
        # 레이블을 정수형으로 변환하여 메트릭 계산
        int_labels = batch['labels'].view(-1).int()
        self.val_acc(similarity_score, int_labels)
        self.val_f1(similarity_score, int_labels)

        batch_size = batch['labels'].size(0)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # dev 데이터셋의 경우 레이블이 있으므로, 임계값 성능을 평가할 수 있습니다.
        if 'labels' in batch:
            embedding1, embedding2 = self(batch['s1_token_ids'], batch['s1_attention_mask'], batch['s2_token_ids'], batch['s2_attention_mask'])
        # test 데이터셋의 경우 레이블이 없습니다.
        else:
            embedding1, embedding2 = self(batch['s1_token_ids'], batch['s1_attention_mask'], batch['s2_token_ids'], batch['s2_attention_mask'])

        distance = F.pairwise_distance(embedding1, embedding2)
        # 거리가 margin/2 보다 작으면 유사(1), 크면 비유사(0)로 예측
        # 이 임계값(0.5)은 실험을 통해 조정할 수 있습니다.
        preds = (distance < self.hparams.margin / 2).long()

        output = {"sent_ids": batch['sent_ids'], "prediction": preds, "dataset_idx": dataloader_idx}
        self.test_outputs.append(output)

    def on_test_epoch_end(self):
        # (이전 코드와 동일)
        output_dir = Path(self.hparams.output_dir) / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        dev_out_path = output_dir / "para-dev-output.csv"
        test_out_path = output_dir / "para-test-output.csv"
        with open(dev_out_path, "w", newline='', encoding='utf-8') as f_dev, \
             open(test_out_path, "w", newline='', encoding='utf-8') as f_test:
            writer_dev = csv.writer(f_dev)
            writer_test = csv.writer(f_test)
            writer_dev.writerow(["id", "Predicted_Is_Paraphrase"])
            writer_test.writerow(["id", "Predicted_Is_Paraphrase"])
            for output in self.test_outputs:
                writer = writer_dev if output['dataset_idx'] == 0 else writer_test
                for sid, pred in zip(output['sent_ids'], output['prediction'].tolist()):
                    writer.writerow([sid, pred])
        print(f"\nDev predictions saved to {dev_out_path}")
        print(f"Test predictions saved to {test_out_path}")
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.0
        )
        return optimizer

# =================================================================================
# 6. 로거 콜백 (변경 없음)
# =================================================================================

class TextLoggerCallback(Callback):
    """매 에포크의 검증 결과를 txt 파일과 python logging 모듈로 동시에 기록하는 콜백"""
    def __init__(self, log_path: Path, hparams: argparse.Namespace, logger: logging.Logger):
        super().__init__()
        self.log_path = log_path
        self.logger = logger
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("# Hyperparameters\n")
            for k, v in sorted(vars(hparams).items()):
                f.write(f"{k}: {v}\n")
            f.write("\n# Metrics per epoch\n")
            f.write("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\tval_f1\n")
        self.logger.info(f"Text log will be saved to {self.log_path}")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        metrics = trainer.logged_metrics
        epoch = pl_module.current_epoch
        train_loss = metrics.get('train/loss_epoch', 'N/A')
        train_acc = metrics.get('train/acc', 'N/A') # train_acc는 on_epoch=True로 설정됨
        val_loss = metrics.get('val/loss', 'N/A')
        val_acc = metrics.get('val/acc', 'N/A')
        val_f1 = metrics.get('val/f1', 'N/A')
        def _fmt(value):
            if isinstance(value, torch.Tensor): value = value.item()
            if isinstance(value, (float, int)): return f"{value:.4f}"
            return str(value)
        log_line = (f"{epoch}\t{_fmt(train_loss)}\t{_fmt(train_acc)}\t"
                    f"{_fmt(val_loss)}\t{_fmt(val_acc)}\t{_fmt(val_f1)}\n")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line)
        self.logger.info(
            f"Epoch {epoch} | train_loss={_fmt(train_loss)}, train_acc={_fmt(train_acc)}, "
            f"val_loss={_fmt(val_loss)}, val_acc={_fmt(val_acc)}, val_f1={_fmt(val_f1)}"
        )

class StepLoggerCallback(Callback):
    """학습 스텝마다(사용자 지정 간격) 손실 및 진행률을 txt 파일 + python logging 으로 기록"""
    def __init__(self, log_path: Path, logger: logging.Logger, log_every_n_steps: int = 50):
        super().__init__()
        self.log_path = log_path
        self.log_every_n_steps = log_every_n_steps
        self.logger = logger
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("global_step\tepoch\tstep_in_epoch\ttimestamp\tloss\n")
        self.logger.info(f"Step log will be saved to {self.log_path}")

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step == 0 or (global_step % self.log_every_n_steps != 0):
            return
        loss = None
        if outputs is not None:
            if isinstance(outputs, torch.Tensor):
                loss = outputs.detach().item()
            elif isinstance(outputs, dict) and "loss" in outputs:
                raw_loss = outputs["loss"]
                loss = raw_loss.detach().item() if isinstance(raw_loss, torch.Tensor) else raw_loss
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        epoch, step_in_epoch = trainer.current_epoch, batch_idx
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = (f"{global_step}\t{epoch}\t{step_in_epoch}\t{timestamp}\t{loss_str}\n")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line)
        self.logger.info(f"Step {global_step} | epoch {epoch} step_in_epoch {step_in_epoch} | loss={loss_str}")

# =================================================================================
# 7. 메인 실행 스크립트 (⭐️ 수정됨: margin 인자 추가)
# =================================================================================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paraphrase Detection with GPT-2 using Contrastive Learning")
    
    # 경로 관련
    parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs_contrastive")

    # 모델 및 학습 하이퍼파라미터
    parser.add_argument("--model_size", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument("--pooling_method", type=str, default='last', choices=['last', 'mean', 'cls'], help="Pooling method for sentence embedding")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-5)
    # --- ⭐️ 대조 손실을 위한 margin 인자 추가 ---
    parser.add_argument("--margin", type=float, default=2.0, help="Margin for Contrastive Loss")
    
    # 시스템 관련
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--accelerator", type=str, default="auto", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--devices", type=int, default=1)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    seed_everything(args.seed)

    output_path = Path(args.output_dir)
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model_size}_{args.pooling_method}_margin{args.margin}"
    run_log_dir = output_path / "logs" / run_name
    run_txt_dir = output_path / "logs_txt"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_txt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_txt_dir)
    logger.info("하이퍼파라미터:")
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}: {v}")

    datamodule = ParaphraseDataModule(args)
    model = ParaphraseLitModule(hparams=args)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints" / run_name,
        filename="best-{epoch}-{val_acc:.4f}",
        save_top_k=3,
        monitor="val/acc",
        mode="max",
        save_last=True
    )
    
    epoch_log_path = run_txt_dir / f"{run_name}_epoch_log.txt"
    text_logger_callback = TextLoggerCallback(log_path=epoch_log_path, hparams=args, logger=logger)
    
    step_log_path = run_txt_dir / f"{run_name}_step_log.txt"
    step_logger_callback = StepLoggerCallback(log_path=step_log_path, logger=logger, log_every_n_steps=50)

    tb_logger = TensorBoardLogger(save_dir=output_path / "logs", name=run_name, version="")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision if args.accelerator == 'gpu' else 32,
        callbacks=[checkpoint_callback, text_logger_callback, step_logger_callback],
        logger=tb_logger,
        log_every_n_steps=50
    )
    
    print("\n--- 대조 학습 훈련을 시작합니다 ---")
    trainer.fit(model, datamodule)
    
    print("\n--- 테스트를 시작합니다 ---")
    trainer.test(datamodule=datamodule, ckpt_path='best')
    
    print("\n--- 파이프라인이 종료되었습니다 ---")
    print(f"결과는 다음 경로에서 확인하세요: {output_path.resolve()}")
    print(f"  - TensorBoard/Text 로그: {run_log_dir.resolve()}")
    print(f"  - TXT 로그: {run_txt_dir.resolve()}")
    print(f"  - 모델 체크포인트: {output_path / 'checkpoints' / run_name}")
    print(f"  - 예측 결과: {output_path / 'predictions'}")

if __name__ == "__main__":
    main()