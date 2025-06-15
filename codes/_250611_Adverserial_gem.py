import argparse
import datetime
import math
import os
import random
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
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

# --- 로컬 모듈 임포트 ---
# [Collaborator] 및 [PreviousCode]의 유틸리티와 클래스를 가져옵니다.
from datasets import (ParaphraseDetectionDataset,
                      ParaphraseDetectionTestDataset, load_paraphrase_data)
# [BasicCode]에서 사용된 모델 구조를 가져옵니다.
from models.gpt2 import GPT2Model

# TQDM 로딩 바 비활성화 여부
TQDM_DISABLE = False


# ---------------------------------------------------------------------------
# 0. 유틸리티 함수 및 클래스
# ---------------------------------------------------------------------------

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
    """
    [ReferenceCode]에서 영감을 받은 텍스트 파일 로거.
    학습 진행 상황을 텍스트 파일에 기록합니다.
    """

    def __init__(self, path: Union[str, Path], step_interval: int = 50):
        super().__init__()
        self.file = Path(path)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.step_interval = step_interval
        self._start_time = None

    def _write(self, msg: str):
        with self.file.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def on_fit_start(self, trainer, pl_module):
        self._start_time = datetime.datetime.now()
        n_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self._write(f"[START] {self._start_time.isoformat()}")
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
        metrics = {k: v.item() for k, v in trainer.callback_metrics.items() if k.startswith(stage)}
        metric_str = " | ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items())
        self._write(f"[{stage.upper():<5}] EPOCH {trainer.current_epoch} END: {metric_str}")

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_epoch_end("train", trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_epoch_end("val", trainer)

    def on_test_epoch_end(self, trainer, pl_module):
        self._write(f"[TEST] Test finished at {datetime.datetime.now().isoformat()}")


# ---------------------------------------------------------------------------
# 1. PyTorch Lightning 데이터 모듈
# ---------------------------------------------------------------------------

class ParaphraseDataModule(pl.LightningDataModule):
    """
    [PreviousCode]의 데이터 모듈을 기반으로, Quora 데이터셋을 로드하고 처리합니다.
    """

    def __init__(self, args):
        super().__init__()
        # hparams에 인자들을 저장하여 LightningModule에서 접근 가능하도록 합니다.
        self.save_hyperparameters(args)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        if stage in ("fit", None):
            train_raw = load_paraphrase_data(self.hparams.para_train, split='train')
            dev_raw = load_paraphrase_data(self.hparams.para_dev, split='dev')
            self.train_set = ParaphraseDetectionDataset(train_raw, self.hparams)
            self.val_set = ParaphraseDetectionDataset(dev_raw, self.hparams)

        if stage in ("test", None):
            # 테스트 시에는 dev와 test 데이터셋을 모두 로드하여 평가합니다.
            dev_raw = load_paraphrase_data(self.hparams.para_dev, split='dev')
            test_raw = load_paraphrase_data(self.hparams.para_test, split='test')
            self.dev_set = ParaphraseDetectionDataset(dev_raw, self.hparams)
            self.test_set = ParaphraseDetectionTestDataset(test_raw, self.hparams)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.train_set.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.val_set.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        # 테스트 시에는 dev와 test 데이터 로더를 리스트로 반환합니다.
        return [
            DataLoader(
                self.dev_set,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                collate_fn=self.dev_set.collate_fn,
                pin_memory=True,
            ),
            DataLoader(
                self.test_set,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                collate_fn=self.test_set.collate_fn,
                pin_memory=True,
            )
        ]


# ---------------------------------------------------------------------------
# 2. PyTorch Lightning 주 모델
# ---------------------------------------------------------------------------

class ParaphraseLitModuleAPT(pl.LightningModule):
    """
    논문의 Adversarial Paraphrasing Task (APT) 원칙을 적용한 Paraphrase Detection 모델.
    - [BasicCode]의 GPT-2 아키텍처를 기반으로 합니다.
    - [PreviousCode]의 R-Drop을 사용하여 모델의 강건성을 높입니다. 이는 피상적인
      어휘 단서에 과적합되지 않도록 하는 APT의 목표와 일치합니다.
    - SimCSE는 제거되었습니다.
    - [ReferenceCode]의 파이프라인 구조(로깅, 콜백)를 따릅니다.
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # 모델 아키텍처 초기화
        self._build_model()

        # 평가 지표 초기화
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")

        # 테스트 결과 저장을 위한 버퍼
        self.test_outputs = []

    def _build_model(self):
        """모델 구성 요소를 초기화합니다."""
        # GPT-2 모델 로드 ([BasicCode] 및 [Collaborator/gpt2.py] 참조)
        # 모델 크기에 따른 hidden_size 자동 설정
        model_config_map = {'gpt2': 768, 'gpt2-medium': 1024, 'gpt2-large': 1280}
        d = model_config_map[self.hparams.model_size]
        
        self.gpt = GPT2Model.from_pretrained(
            self.hparams.model_size, d=d, l=12, num_heads=12
        )
        self.classification_head = nn.Linear(d, 2)

        # 모델의 모든 파라미터를 fine-tuning 대상으로 설정
        for param in self.parameters():
            param.requires_grad = True

    def _get_representation(self, input_ids, attention_mask):
        """입력에서 GPT-2 표현과 분류 로짓을 추출합니다."""
        # [Collaborator/gpt2.py]의 forward와 유사
        gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        # 마지막 토큰의 hidden state 사용 ([BasicCode] forward 참조)
        last_token_hidden_state = gpt_out['last_token']
        logits = self.classification_head(last_token_hidden_state)
        return logits

    def forward(self, input_ids, attention_mask):
        return self._get_representation(input_ids, attention_mask)

    def _compute_apt_loss(self, logits1, logits2, labels):
        """
        Adversarial Paraphrasing Task (APT)의 원칙을 반영한 손실 함수.
        - CE Loss: 기본적인 분류 손실
        - R-Drop (KL-Divergence): 모델의 강건성을 높여 어휘적 단서에 대한 의존도를 줄임
        """
        # 1. Cross-Entropy Loss (with Label Smoothing)
        ce_loss_1 = F.cross_entropy(logits1, labels, label_smoothing=self.hparams.label_smoothing)
        ce_loss_2 = F.cross_entropy(logits2, labels, label_smoothing=self.hparams.label_smoothing)
        ce_loss = (ce_loss_1 + ce_loss_2) / 2

        # 2. R-Drop Loss (KL-Divergence)
        kl_div = F.kl_div(
            F.log_softmax(logits1, dim=-1),
            F.softmax(logits2, dim=-1),
            reduction='batchmean'
        )
        kl_loss = self.hparams.rdrop_alpha * kl_div

        total_loss = ce_loss + kl_loss
        return total_loss, ce_loss, kl_loss

    def training_step(self, batch, batch_idx):
        ids, mask, labels = batch["token_ids"], batch["attention_mask"], batch["labels"].flatten()

        # R-Drop을 위해 동일한 입력에 대해 두 번의 forward pass 수행 (dropout이 다르게 적용됨)
        logits1 = self._get_representation(ids, mask)
        logits2 = self._get_representation(ids, mask)

        # 손실 계산
        total_loss, ce_loss, kl_loss = self._compute_apt_loss(logits1, logits2, labels)

        # 지표 로깅
        self.log_dict({
            "train/loss": total_loss,
            "train/ce_loss": ce_loss,
            "train/kl_loss": kl_loss,
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=ids.size(0))

        # 정확도 계산 및 로깅
        preds = torch.argmax(logits1, dim=1)
        self.train_acc(preds, labels)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=ids.size(0))

        return total_loss
        
    def validation_step(self, batch, batch_idx):
        ids, mask, labels = batch["token_ids"], batch["attention_mask"], batch["labels"].flatten()

        # 검증 시에는 단일 forward pass 수행
        logits = self._get_representation(ids, mask)
        loss = F.cross_entropy(logits, labels)

        # 지표 계산
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)

        # 지표 로깅
        self.log_dict({
            "val/loss": loss,
            "val/acc": self.val_acc,
            "val/f1": self.val_f1,
        }, on_epoch=True, prog_bar=True, logger=True, batch_size=ids.size(0))
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # dataloader_idx: 0 for dev, 1 for test
        ids, mask, sent_ids = batch["token_ids"], batch["attention_mask"], batch["sent_ids"]

        logits = self._get_representation(ids, mask)
        preds = torch.argmax(logits, dim=1)

        # 결과 저장
        for sid, pred in zip(sent_ids, preds.tolist()):
            self.test_outputs.append({
                "sent_id": sid,
                "prediction": pred,
                "dataset_idx": dataloader_idx
            })

    def on_test_epoch_end(self):
        # [BasicCode]의 test 함수와 같이 예측 파일을 생성합니다.
        os.makedirs("predictions", exist_ok=True)
        
        dev_preds = [o for o in self.test_outputs if o['dataset_idx'] == 0]
        test_preds = [o for o in self.test_outputs if o['dataset_idx'] == 1]

        # dev 결과 파일 작성
        with open(self.hparams.para_dev_out, "w+") as f:
            f.write("id\tPredicted_Is_Paraphrase\n")
            for item in dev_preds:
                f.write(f"{item['sent_id']}\t{item['prediction']}\n")
        print(f"Dev predictions saved to {self.hparams.para_dev_out}")

        # test 결과 파일 작성
        with open(self.hparams.para_test_out, "w+") as f:
            f.write("id\tPredicted_Is_Paraphrase\n")
            for item in test_preds:
                f.write(f"{item['sent_id']}\t{item['prediction']}\n")
        print(f"Test predictions saved to {self.hparams.para_test_out}")
        
        # 버퍼 비우기
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        
        # Linear Warmup Scheduler 설정
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * self.hparams.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

# ---------------------------------------------------------------------------
# 3. 메인 실행 스크립트
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Paraphrase Detection Training Pipeline with APT principles")
    
    # 데이터 경로
    parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
    
    # 출력 경로
    parser.add_argument("--para_dev_out", type=str, default="../predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="../predictions/para-test-output.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs")

    # 모델 및 학습 하이퍼파라미터
    parser.add_argument("--model_size", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length for tokenizer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--rdrop_alpha", type=float, default=4.0, help="Weight for R-Drop KL loss")
    
    # 시스템 설정
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["16-mixed", "32-true"])
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")

    return parser.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)

    # 출력 디렉토리 생성
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 시간 기반으로 실행 이름 생성
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model_size} + Adversarial_GEM"

    # 데이터 모듈 초기화
    datamodule = ParaphraseDataModule(args)

    # 모델 초기화
    model = ParaphraseLitModuleAPT(hparams=args)
    
    # 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints" / run_name,
        filename="best-{epoch}-{val/acc:.4f}",
        save_top_k=3,
        monitor="val/acc",
        mode="max",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # 로거 설정
    tb_logger = TensorBoardLogger(output_path / "logs", name=run_name)
    txt_logger = TextFileLogger(output_path / "logs" / run_name / "train_log.txt")

    # 트레이너 초기화
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor, txt_logger],
        logger=tb_logger,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # 학습 시작
    print("--- Starting Training ---")
    trainer.fit(model, datamodule)
    
    # 테스트 시작
    print("\n--- Starting Testing ---")
    # `ckpt_path='best'`를 사용하여 최상의 체크포인트로 테스트
    trainer.test(model, datamodule, ckpt_path='best')
    
    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    main()