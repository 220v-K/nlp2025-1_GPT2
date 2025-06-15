# main_pl_final.py
import argparse
import csv
import datetime
import os
import random
import logging
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
from transformers import GPT2Tokenizer, GPT2Model as HFGPT2Model

#! 주의: 메모리 문제 발생 시 주석 처리
torch.set_float32_matmul_precision('high')

# =================================================================================
# 1. 설정 및 유틸리티 함수
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

# ---------------------------------------------------------------------------------
# 로깅 설정 함수 (기존 _250523_default_paraphrase.py 방식을 차용)
# ---------------------------------------------------------------------------------

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

# =================================================================================
# 2. 데이터셋 클래스 (변경 없음)
# =================================================================================

def load_paraphrase_data(paraphrase_filename: str, split: str = 'train'):
    """Quora 데이터셋을 로드합니다."""
    data = []
    # ... (이전 코드와 동일)
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
# 4. PyTorch Lightning 모델 (⭐️ 수정됨: 풀링 레이어 선택 기능 추가)
# =================================================================================

class ParaphraseLitModule(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.gpt = HFGPT2Model.from_pretrained(self.hparams.model_size)
        self.classification_head = nn.Linear(self.gpt.config.hidden_size, 2)
        # 모든 파라미터 학습
        for param in self.parameters():
            param.requires_grad = True
        
        # Metric 정의
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
        self.test_outputs = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        선택된 풀링 방법에 따라 문장 표현을 추출하고 분류를 수행합니다.
        """
        gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = gpt_out.last_hidden_state

        pooled_output = None
        # --- 👇 풀링 방법 선택 로직 ---
        if self.hparams.pooling_method == 'last':
            # 기존 방식: 마지막 토큰의 hidden state 사용
            last_idx = attention_mask.sum(dim=1) - 1
            pooled_output = hidden_states[torch.arange(hidden_states.size(0)), last_idx]
        
        elif self.hparams.pooling_method == 'mean':
            # Mean Pooling: attention mask를 이용하여 패딩을 제외한 토큰들의 평균을 계산
            expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden_states = torch.sum(hidden_states * expanded_mask, dim=1)
            sum_mask = torch.sum(expanded_mask, dim=1)
            # 0으로 나누는 것을 방지
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_hidden_states / sum_mask

        elif self.hparams.pooling_method == 'cls':
            # CLS Pooling: 첫 번째 토큰의 hidden state를 문장 표현으로 사용
            pooled_output = hidden_states[:, 0, :]

        else:
            raise ValueError("Unsupported pooling method. Choose from 'last', 'mean', 'cls'.")
        # --- 👆 여기까지 ---

        logits = self.classification_head(pooled_output)
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
        # ... (이전 코드와 동일)
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
# 5. 새로운 기능: 텍스트 로거 콜백
# =================================================================================

class TextLoggerCallback(Callback):
    """매 에포크의 검증 결과를 txt 파일과 python logging 모듈로 동시에 기록하는 콜백"""

    def __init__(self, log_path: Path, hparams: argparse.Namespace, logger: logging.Logger):
        super().__init__()
        self.log_path = log_path
        self.logger = logger

        # 파일 초기화 및 하이퍼파라미터 기록
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("# Hyperparameters\n")
            for k, v in sorted(vars(hparams).items()):
                f.write(f"{k}: {v}\n")
            f.write("\n# Metrics per epoch\n")
            f.write("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\tval_f1\n")

        self.logger.info(f"Text log will be saved to {self.log_path}")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """검증 에포크가 끝날 때 호출됩니다."""
        metrics = trainer.logged_metrics
        epoch = pl_module.current_epoch
        
        # 메트릭 값 추출 (존재하지 않을 경우 'N/A'로 처리)
        train_loss = metrics.get('train/loss_epoch', 'N/A')
        train_acc = metrics.get('train/acc_epoch', 'N/A')
        val_loss = metrics.get('val/loss', 'N/A')
        val_acc = metrics.get('val/acc', 'N/A')
        val_f1 = metrics.get('val/f1', 'N/A')

        # 메트릭 문자열 포맷 함수 (숫자일 경우 소수점 4자리, 그 외는 문자열 그대로)
        def _fmt(value):
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (float, int)):
                return f"{value:.4f}"
            return str(value)

        log_line = (
            f"{epoch}\t"
            f"{_fmt(train_loss)}\t"
            f"{_fmt(train_acc)}\t"
            f"{_fmt(val_loss)}\t"
            f"{_fmt(val_acc)}\t"
            f"{_fmt(val_f1)}\n"
        )

        # 파일 기록
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

        # 로거에도 기록
        self.logger.info(
            f"Epoch {epoch} | train_loss={_fmt(train_loss)}, train_acc={_fmt(train_acc)}, "
            f"val_loss={_fmt(val_loss)}, val_acc={_fmt(val_acc)}, val_f1={_fmt(val_f1)}"
        )

# =================================================================================
# 5-1. 새로운 기능: 스텝 로거 콜백 (학습 진행 상황을 주기적으로 기록)
# =================================================================================

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
        """지정된 간격마다 현재 스텝 정보를 로그에 기록"""
        global_step = trainer.global_step  # 전체 스텝 번호 (0부터 시작)
        if global_step == 0 or (global_step % self.log_every_n_steps != 0):
            return

        # 출력 텐서가 있는 경우 손실값을 스칼라로 변환
        loss = None
        if outputs is not None:
            if isinstance(outputs, torch.Tensor):
                loss = outputs.detach().item()
            elif isinstance(outputs, dict) and "loss" in outputs:
                raw_loss = outputs["loss"]
                loss = raw_loss.detach().item() if isinstance(raw_loss, torch.Tensor) else raw_loss
        # 손실을 알 수 없으면 'N/A'
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"

        epoch = trainer.current_epoch
        step_in_epoch = batch_idx  # 0-based 인덱스
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_line = (
            f"{global_step}\t{epoch}\t{step_in_epoch}\t{timestamp}\t{loss_str}\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

        # 로거에도 기록
        self.logger.info(
            f"Step {global_step} | epoch {epoch} step_in_epoch {step_in_epoch} | loss={loss_str}"
        )

# =================================================================================
# 6. 메인 실행 스크립트 (⭐️ 수정됨: 인자 추가 및 콜백 등록)
# =================================================================================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paraphrase Detection with GPT-2 using PyTorch Lightning")
    
    # 경로 관련
    parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs")

    # 모델 및 학습 하이퍼파라미터
    parser.add_argument("--model_size", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    # --- 풀링 방법 선택 인자 추가 ---
    parser.add_argument("--pooling_method", type=str, default='last', choices=['last', 'mean', 'cls'], help="Pooling method for classification head")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=96)
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
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model_size}_{args.pooling_method}"
    run_log_dir = output_path / "logs" / run_name
    run_txt_dir = output_path / "logs_txt"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_txt_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 로깅 설정 및 하이퍼파라미터 기록
    # -----------------------------------------------------
    logger = setup_logging(run_txt_dir)

    logger.info("하이퍼파라미터:")
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}: {v}")

    datamodule = ParaphraseDataModule(args)
    model = ParaphraseLitModule(hparams=args)
    
    # 체크포인트 콜백
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints" / run_name,
        filename="best-{epoch}-{val_acc:.4f}",
        save_top_k=3,
        monitor="val/acc",
        mode="max",
        save_last=True
    )
    
    # --- 👇 텍스트 로거 콜백(에포크 단위) 생성 ---
    epoch_log_path = run_txt_dir / f"{run_name}_epoch_log.txt"
    text_logger_callback = TextLoggerCallback(log_path=epoch_log_path, hparams=args, logger=logger)
    
    # --- 👇 스텝 로거 콜백(스텝 단위) 생성 ---
    step_log_path = run_txt_dir / f"{run_name}_step_log.txt"
    step_logger_callback = StepLoggerCallback(log_path=step_log_path, logger=logger, log_every_n_steps=50)

    # TensorBoard 로거
    tb_logger = TensorBoardLogger(save_dir=output_path / "logs", name=run_name, version="")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision if args.accelerator == 'gpu' else 32,
        # --- 👇 콜백 리스트에 텍스트 로거 추가 ---
        callbacks=[checkpoint_callback, text_logger_callback, step_logger_callback],
        logger=tb_logger,
        log_every_n_steps=50
    )
    
    print("\n--- 훈련을 시작합니다 ---")
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