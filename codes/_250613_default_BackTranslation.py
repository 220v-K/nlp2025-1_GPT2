# main_pl_strict.py
import argparse
import csv
import datetime
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2Model as HFGPT2Model

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

# =================================================================================
# 2. 데이터셋 클래스 (변경 없음)
# =================================================================================

def load_paraphrase_data(paraphrase_filename: str, split: str = 'train'):
    """Quora 데이터셋을 로드합니다. 원본 로직을 따릅니다."""
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

class ParaphraseDataset(Dataset):
    def __init__(self, dataset: list, tokenizer: GPT2Tokenizer, max_length: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        s1 = item['sentence1']
        s2 = item['sentence2']
        label = item['is_duplicate']
        sent_id = item['id']
        prompt = f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '
        encoding = self.tokenizer(
            prompt, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'token_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'sent_ids': sent_id
        }

class ParaphraseTestDataset(Dataset):
    def __init__(self, dataset: list, tokenizer: GPT2Tokenizer, max_length: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        s1 = item['sentence1']
        s2 = item['sentence2']
        sent_id = item['id']
        prompt = f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '
        encoding = self.tokenizer(
            prompt, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'token_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
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
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)
    def test_dataloader(self):
        return [
            DataLoader(self.dev_for_test_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True),
            DataLoader(self.test_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)
        ]

# =================================================================================
# 4. PyTorch Lightning 모델 (수정됨)
# =================================================================================

class ParaphraseLitModule(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.gpt = HFGPT2Model.from_pretrained(self.hparams.model_size)
        self.classification_head = nn.Linear(self.gpt.config.hidden_size, 2)
        for param in self.parameters():
            param.requires_grad = True
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
        self.test_outputs = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = gpt_out.last_hidden_state
        last_idx = attention_mask.sum(dim=1) - 1
        last_token_state = hidden_states[torch.arange(hidden_states.size(0)), last_idx]
        logits = self.classification_head(last_token_state)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch['token_ids'], batch['attention_mask'])
        loss = F.cross_entropy(logits, batch['labels'])
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, batch['labels'])
        batch_size = batch['labels'].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['token_ids'], batch['attention_mask'])
        loss = F.cross_entropy(logits, batch['labels'])
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, batch['labels'])
        self.val_f1(preds, batch['labels'])
        batch_size = batch['labels'].size(0)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch['token_ids'], batch['attention_mask'])
        preds = torch.argmax(logits, dim=1)
        output = {"sent_ids": batch['sent_ids'], "prediction": preds, "dataset_idx": dataloader_idx}
        self.test_outputs.append(output)

    def on_test_epoch_end(self):
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
        """
        [수정됨] 원본 코드와 동일하게 스케줄러 없이 AdamW 옵티마이저만 사용합니다.
        weight_decay도 원본과 같이 0.0으로 설정합니다.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=0.0
        )
        return optimizer

# =================================================================================
# 5. 메인 실행 스크립트 (수정됨)
# =================================================================================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paraphrase Detection with GPT-2 using PyTorch Lightning")
    
    # 경로 관련
    parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs")

    # 모델 및 학습 하이퍼파라미터 (원본 코드 기준)
    parser.add_argument("--model_size", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=96) # [수정됨] 원본 기본값 96으로 변경
    parser.add_argument("--lr", type=float, default=2e-5)
    
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
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model_size}_baseline_strict"

    datamodule = ParaphraseDataModule(args)
    model = ParaphraseLitModule(hparams=args)
    
    # [수정됨] LearningRateMonitor 콜백 제거
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints" / run_name,
        filename="best-{epoch}-{val_acc:.4f}",
        save_top_k=3,
        monitor="val/acc",
        mode="max",
        save_last=True
    )
    
    tb_logger = TensorBoardLogger(output_path / "logs", name=run_name)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision if args.accelerator == 'gpu' else 32,
        callbacks=[checkpoint_callback], # [수정됨] lr_monitor 제거
        logger=tb_logger,
        log_every_n_steps=50
    )
    
    print("\n--- 훈련을 시작합니다 ---")
    trainer.fit(model, datamodule)
    
    print("\n--- 테스트를 시작합니다 ---")
    trainer.test(datamodule=datamodule, ckpt_path='best')
    
    print("\n--- 파이프라인이 종료되었습니다 ---")
    print(f"결과는 다음 경로에서 확인하세요: {output_path.resolve()}")
    print(f"  - TensorBoard 로그: {output_path / 'logs' / run_name}")
    print(f"  - 모델 체크포인트: {output_path / 'checkpoints' / run_name}")
    print(f"  - 예측 결과: {output_path / 'predictions'}")

if __name__ == "__main__":
    main()