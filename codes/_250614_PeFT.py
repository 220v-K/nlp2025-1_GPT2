# python _250614_PeFT.py --model_size gpt2 --epochs 20 --batch_size 128 --finetune_method lora --lora_r 16 --lora_alpha 32
# python _250614_PeFT.py --model_size gpt2 --epochs 20 --batch_size 128 --finetune_method dora --lora_r 16 --lora_alpha 32


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
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification as HFGPT2ForSeqCls
# [추가] PEFT 라이브러리 임포트
from peft import get_peft_model, LoraConfig, TaskType

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
# 4. [신규] 텍스트 로깅 콜백
# =================================================================================
class TextLoggerCallback(Callback):
    def __init__(self, log_dir: str, run_name: str):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.log_file_path = self.log_dir / f"{self.run_name}_log.txt"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = None
    
    def _log(self, message: str):
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.start_time = datetime.datetime.now()
        self._log("="*50)
        self._log(f"Run Name: {self.run_name}")
        self._log(f"Training started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("="*50)
        self._log("\n--- Hyperparameters ---")
        # hparams는 Namespace 또는 AttributeDict(매핑)일 수 있음
        hparams_obj = pl_module.hparams
        try:
            items = hparams_obj.items()  # Dict-like 객체 처리
        except AttributeError:
            items = vars(hparams_obj).items()  # argparse.Namespace 등 처리

        for key, value in items:
            self._log(f"{key}: {value}")
        self._log("-" * 25 + "\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % trainer.log_every_n_steps == 0:
            loss = outputs['loss'].item()
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_msg = f"Time: {current_time} | Step: {trainer.global_step + 1} | Train Loss: {loss:.4f}"
            self._log(log_msg)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val/loss', 'N/A')
        val_acc = metrics.get('val/acc', 'N/A')
        val_f1 = metrics.get('val/f1', 'N/A')
        
        # torch.Tensor를 float로 변환
        if hasattr(val_loss, 'item'): val_loss = val_loss.item()
        if hasattr(val_acc, 'item'): val_acc = val_acc.item()
        if hasattr(val_f1, 'item'): val_f1 = val_f1.item()
        
        self._log("-" * 25)
        self._log(f"Time: {current_time} | *** Validation Epoch {epoch} End ***")
        self._log(f"  - Val Loss: {val_loss:.4f}")
        self._log(f"  - Val Acc:  {val_acc:.4f}")
        self._log(f"  - Val F1:   {val_f1:.4f}")
        self._log("-" * 25)
        
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self._log("\n--- Test Results (on dev set) ---")
        metrics = trainer.callback_metrics
        test_loss = metrics.get('test/loss_dev', 'N/A')
        test_acc = metrics.get('test/acc_dev', 'N/A')
        test_f1 = metrics.get('test/f1_dev', 'N/A')

        if hasattr(test_loss, 'item'): test_loss = test_loss.item()
        if hasattr(test_acc, 'item'): test_acc = test_acc.item()
        if hasattr(test_f1, 'item'): test_f1 = test_f1.item()
        
        self._log(f"  - Test Loss (dev): {test_loss:.4f}")
        self._log(f"  - Test Acc (dev):  {test_acc:.4f}")
        self._log(f"  - Test F1 (dev):   {test_f1:.4f}")
    
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        self._log("\n" + "="*50)
        self._log(f"Training finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Total training duration: {duration}")
        self._log("="*50)

# =================================================================================
# 5. PyTorch Lightning 모델 (수정됨)
# =================================================================================

class ParaphraseLitModule(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # GPT2 시퀀스 분류 모델 로드 (labels=2)
        self.gpt = HFGPT2ForSeqCls.from_pretrained(self.hparams.model_size, num_labels=2)
        # padding token이 정의되어 있지 않으면 eos 토큰을 패딩 토큰으로 설정
        if self.gpt.config.pad_token_id is None:
            self.gpt.config.pad_token_id = self.gpt.config.eos_token_id
        
        # [수정] 파인튜닝 방식에 따라 모델 구성 변경
        if self.hparams.finetune_method in ['lora', 'dora']:
            print(f"Applying {self.hparams.finetune_method.upper()} fine-tuning...")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=["c_attn"], # GPT-2의 Attention 가중치를 타겟으로 지정
                use_dora=(self.hparams.finetune_method == 'dora')
            )
            self.gpt = get_peft_model(self.gpt, peft_config)
            self.gpt.print_trainable_parameters()
        else:  # 'full' fine-tuning
            print("Applying Full fine-tuning...")
            for param in self.gpt.parameters():
                param.requires_grad = True

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
        self.test_acc_dev = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_f1_dev = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
        
        self.test_outputs = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

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
        
        # [수정] dataloader_idx=0은 dev set이므로 성능 측정
        if dataloader_idx == 0:
            loss = F.cross_entropy(logits, batch['labels'])
            self.test_acc_dev(preds, batch['labels'])
            self.test_f1_dev(preds, batch['labels'])
            batch_size = batch['labels'].size(0)
            self.log("test/loss_dev", loss, on_epoch=True, logger=True, batch_size=batch_size)
            self.log("test/acc_dev", self.test_acc_dev, on_epoch=True, logger=True, batch_size=batch_size)
            self.log("test/f1_dev", self.test_f1_dev, on_epoch=True, logger=True, batch_size=batch_size)

        output = {"sent_ids": batch['sent_ids'], "prediction": preds, "dataset_idx": dataloader_idx}
        self.test_outputs.append(output)

    def on_test_epoch_end(self):
        # [수정] 예측 파일 저장 전에 최종 test metric 출력
        print("\n--- Test Metrics (on dev set) ---")
        final_acc = self.test_acc_dev.compute()
        final_f1 = self.test_f1_dev.compute()
        print(f"  - Test Accuracy (dev): {final_acc:.4f}")
        print(f"  - Test F1-score (dev): {final_f1:.4f}")
        
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
        PEFT를 사용하면 `self.parameters()`가 학습 가능한 파라미터(어댑터, 분류 헤드 등)만
        자동으로 반환하므로 옵티마이저 코드는 수정할 필요가 없습니다.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=0.0
        )
        return optimizer

# =================================================================================
# 6. 메인 실행 스크립트 (수정됨)
# =================================================================================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paraphrase Detection with PEFT using PyTorch Lightning")
    
    # 경로 관련
    parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs")

    # 모델 및 학습 하이퍼파라미터
    parser.add_argument("--model_size", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-5)
    
    # [추가] 파인튜닝 방식 선택
    parser.add_argument("--finetune_method", type=str, default="full", choices=["full", "lora", "dora"])
    
    # [추가] LoRA/DoRA 하이퍼파라미터
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension (rank)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

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
    # [수정] 실행 이름에 파인튜닝 방식 추가
    run_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model_size}_{args.finetune_method}"

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
    
    tb_logger = TensorBoardLogger(output_path / "logs", name=run_name)
    
    # [추가] 텍스트 로거 콜백 인스턴스 생성
    text_logger_callback = TextLoggerCallback(log_dir=output_path / "logs_txt", run_name=run_name)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision if args.accelerator == 'gpu' else 32,
        # [수정] 콜백 리스트에 text_logger_callback 추가
        callbacks=[checkpoint_callback, text_logger_callback],
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
    print(f"  - 텍스트 로그: {text_logger_callback.log_file_path}")
    print(f"  - 모델 체크포인트: {output_path / 'checkpoints' / run_name}")
    print(f"  - 예측 결과: {output_path / 'predictions'}")

if __name__ == "__main__":
    main()