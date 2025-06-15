import argparse
import datetime
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup

# --- 제공된 모듈 임포트 ---
# [Collaborator] 및 [BasicCode]의 유틸리티와 클래스를 가져옵니다.
from datasets import (ParaphraseDetectionDataset,
                      ParaphraseDetectionTestDataset, load_paraphrase_data)
from models.gpt2 import GPT2Model  # Base GPT-2 model

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
    [Reference&PreviousCode]에서 영감을 받은 텍스트 파일 로거.
    학습 진행 상황과 최종 테스트 결과를 텍스트 파일에 기록합니다.
    """

    def __init__(self, path: Union[str, Path], step_interval: int = 50):
        super().__init__()
        self.file = Path(path)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        # 로그 파일이 이미 존재하면 삭제하여 새로 시작
        if self.file.exists():
            self.file.unlink()
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
        # 테스트 결과는 pl_module의 on_test_epoch_end에서 직접 기록
        self._write(f"[FINISH] Pipeline finished at {datetime.datetime.now().isoformat()}")


# ---------------------------------------------------------------------------
# 1. MTAN 논문 기반의 새로운 모듈
# ---------------------------------------------------------------------------

class AttentionModule(nn.Module):
    """
    "End-to-End Multi-Task Learning with Attention" 논문 의
    Task-Specific Attention Module을 Single-Task에 맞게 변형한 모듈. 
    
    이 모듈은 입력 피처에 적용할 soft attention mask를 학습합니다. 
    마스크는 입력 피처 자체에 기반하여 동적으로 생성되며, 
    이를 통해 모델이 각 레이어에서 중요한 피처에 집중하도록 합니다. 
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # 논문의 1x1 conv를 시퀀스 데이터에 맞게 Linear 레이어로 구현 
        self.g_linear = nn.Linear(hidden_size, hidden_size)
        self.h_linear = nn.Linear(hidden_size, hidden_size)
        
        # Transformer에서는 BatchNorm 대신 LayerNorm이 더 안정적
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): GPT-2 레이어의 출력 텐서. shape: [B, S, D]

        Returns:
            torch.Tensor: 어텐션이 적용된 피처 텐서. shape: [B, S, D]
        """
        # 마스크 생성을 위한 피처 변환
        mask_gen_features = self.g_linear(features)
        mask_gen_features = self.norm(mask_gen_features)
        mask_gen_features = F.relu(mask_gen_features)
        
        # 최종 어텐션 마스크 계산 (Sigmoid를 통해 0과 1 사이 값으로) 
        attention_mask = torch.sigmoid(self.h_linear(mask_gen_features))
        
        # 입력 피처에 element-wise 곱으로 어텐션 적용 
        attended_features = features * attention_mask
        
        return attended_features


class GPT2WithMTAN(GPT2Model):
    """
    기존 GPT2Model에 MTAN의 AttentionModule을 통합한 모델.
    """
    def __init__(self, config):
        super().__init__(config)
        # 각 GPT-2 레이어 뒤에 적용될 AttentionModule 리스트 생성
        self.attention_modules = nn.ModuleList(
            [AttentionModule(config.hidden_size) for _ in range(config.num_hidden_layers)]
        )

    def encode(self, hidden_states, attention_mask):
        """
        기존 encode 메서드를 오버라이드하여 각 트랜스포머 레이어 뒤에
        어텐션 모듈을 적용합니다.
        """
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, hidden_states.dtype)

        for i, layer_module in enumerate(self.gpt_layers):
            # 1. 원래 GPT-2 레이어를 통과하여 '공유 피처' 생성
            shared_features = layer_module(hidden_states, extended_attention_mask)
            
            # 2. 어텐션 모듈을 통과하여 피처를 정제
            attended_features = self.attention_modules[i](shared_features)
            
            # 3. 다음 레이어의 입력으로 어텐션이 적용된 피처를 사용
            hidden_states = attended_features
            
        return hidden_states


# ---------------------------------------------------------------------------
# 2. PyTorch Lightning 데이터 모듈
# ---------------------------------------------------------------------------

class ParaphraseDataModule(pl.LightningDataModule):
    """
    [Reference&PreviousCode]의 데이터 모듈을 기반으로, Quora 데이터셋을 로드하고 처리합니다.
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # GPT2Tokenizer 초기화 시 pad_token 설정
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage: str = None):
        if stage in ("fit", None):
            train_raw = load_paraphrase_data(self.hparams.para_train, split='train')
            val_raw = load_paraphrase_data(self.hparams.para_dev, split='dev')
            self.train_set = ParaphraseDetectionDataset(train_raw, self.hparams)
            self.val_set = ParaphraseDetectionDataset(val_raw, self.hparams)

        if stage in ("test", None):
            # 테스트 시에는 dev와 test 데이터셋을 모두 평가
            dev_raw = load_paraphrase_data(self.hparams.para_dev, split='dev')
            test_raw = load_paraphrase_data(self.hparams.para_test, split='test')
            self.dev_for_test_set = ParaphraseDetectionDataset(dev_raw, self.hparams)
            self.test_set = ParaphraseDetectionTestDataset(test_raw, self.hparams)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers, collate_fn=self.train_set.collate_fn, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, collate_fn=self.val_set.collate_fn, pin_memory=True
        )

    def test_dataloader(self):
        # dev 셋(레이블 포함)과 test 셋(레이블 미포함)을 리스트로 반환
        return [
            DataLoader(
                self.dev_for_test_set, batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, collate_fn=self.dev_for_test_set.collate_fn, pin_memory=True
            ),
            DataLoader(
                self.test_set, batch_size=self.hparams.batch_size, shuffle=False,
                num_workers=self.hparams.num_workers, collate_fn=self.test_set.collate_fn, pin_memory=True
            )
        ]


# ---------------------------------------------------------------------------
# 3. PyTorch Lightning 주 모델
# ---------------------------------------------------------------------------

class ParaphraseMTANLitModule(pl.LightningModule):
    """
    MTAN 방법론을 적용한 Paraphrase Detection을 위한 PyTorch Lightning 모듈.
    - [BasicCode]의 GPT-2 아키텍처를 기반으로 합니다.
    - SimCSE, R-Drop 등은 제거하고 MTAN의 Attention 메커니즘을 적용합니다.
    - [Reference&PreviousCode]의 파이프라인 구조(로깅, 콜백)를 따릅니다.
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self._build_model()
        self._init_metrics()
        self.test_outputs = []

    def _build_model(self):
        """모델 구성 요소를 초기화합니다."""
        # 모델 크기에 따른 hidden_size 및 기타 파라미터 설정
        model_configs = {
            'gpt2': {'d': 768, 'l': 12, 'num_heads': 12},
            'gpt2-medium': {'d': 1024, 'l': 24, 'num_heads': 16},
            'gpt2-large': {'d': 1280, 'l': 36, 'num_heads': 20}
        }
        config = model_configs[self.hparams.model_size]
        
        # MTAN 어텐션 모듈이 추가된 GPT-2 모델 로드
        self.gpt = GPT2WithMTAN.from_pretrained(
            model=self.hparams.model_size, **config
        )
        self.classification_head = nn.Linear(config['d'], 2)

        # 모델 전체를 fine-tuning
        for param in self.parameters():
            param.requires_grad = True

    def _init_metrics(self):
        """평가 지표를 초기화합니다."""
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_roc = torchmetrics.AUROC(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")

    def forward(self, input_ids, attention_mask):
        """[BasicCode]의 forward 로직과 동일"""
        gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_token_hidden_state = gpt_out['last_token']
        logits = self.classification_head(last_token_hidden_state)
        return logits

    def training_step(self, batch, batch_idx):
        ids, mask, labels = batch["token_ids"], batch["attention_mask"], batch["labels"].flatten()
        logits = self(ids, mask)
        loss = F.cross_entropy(logits, labels)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        ids, mask, labels = batch["token_ids"], batch["attention_mask"], batch["labels"].flatten()
        logits = self(ids, mask)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)[:, 1] # 확률 for ROC-AUC
        
        # 모든 검증 지표 업데이트 및 로깅
        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_roc.update(probs, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)

        self.log_dict({
            "val/loss": loss,
            "val/acc": self.val_acc,
            "val/f1": self.val_f1,
            "val/roc_auc": self.val_roc,
            "val/precision": self.val_precision,
            "val/recall": self.val_recall,
        }, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        ids, mask, sent_ids = batch["token_ids"], batch["attention_mask"], batch["sent_ids"]
        logits = self(ids, mask)
        preds = torch.argmax(logits, dim=1)

        # 개별 예측 결과를 저장
        for i in range(len(sent_ids)):
            output = {
                "sent_id": sent_ids[i],
                "prediction": preds[i].item(),
                "dataset_idx": dataloader_idx
            }
            # dev set (idx=0)은 레이블이 있으므로 함께 저장
            if dataloader_idx == 0:
                output["label"] = batch["labels"].flatten()[i].item()
            self.test_outputs.append(output)

    def on_test_epoch_end(self):
        """테스트 종료 후 예측 파일 생성 및 dev set 성능 평가"""
        output_dir = Path(self.hparams.output_dir)
        pred_dir = output_dir / "predictions"
        pred_dir.mkdir(exist_ok=True, parents=True)

        dev_outputs = [o for o in self.test_outputs if o['dataset_idx'] == 0]
        test_outputs = [o for o in self.test_outputs if o['dataset_idx'] == 1]

        # dev 결과 파일 작성
        dev_out_path = pred_dir / self.hparams.para_dev_out.split('/')[-1]
        with open(dev_out_path, "w+") as f:
            f.write("id\tPredicted_Is_Paraphrase\n")
            for item in dev_outputs:
                f.write(f"{item['sent_id']}\t{item['prediction']}\n")
        print(f"Dev predictions saved to {dev_out_path}")
        
        # test 결과 파일 작성
        test_out_path = pred_dir / self.hparams.para_test_out.split('/')[-1]
        with open(test_out_path, "w+") as f:
            f.write("id\tPredicted_Is_Paraphrase\n")
            for item in test_outputs:
                f.write(f"{item['sent_id']}\t{item['prediction']}\n")
        print(f"Test predictions saved to {test_out_path}")
        
        # Dev set에 대한 최종 성능 지표 계산 및 로깅
        if dev_outputs:
            labels = [o['label'] for o in dev_outputs]
            preds = [o['prediction'] for o in dev_outputs]
            
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='binary')
            precision = precision_score(labels, preds, average='binary')
            recall = recall_score(labels, preds, average='binary')
            
            # 텍스트 로거에 결과 기록
            text_logger = next((cb for cb in self.trainer.callbacks if isinstance(cb, TextFileLogger)), None)
            if text_logger:
                text_logger._write("\n--- FINAL DEV SET EVALUATION ---")
                text_logger._write(f"  - Accuracy:  {acc:.4f}")
                text_logger._write(f"  - F1-Score:  {f1:.4f}")
                text_logger._write(f"  - Precision: {precision:.4f}")
                text_logger._write(f"  - Recall:    {recall:.4f}")
                text_logger._write("------------------------------------")
            
            # 별도 metrics 파일 저장 (predictions 폴더)
            metrics_file = pred_dir / f"dev_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.hparams.model_size}.txt"
            with open(metrics_file, "w", encoding="utf-8") as mf:
                mf.write("Metric\tValue\n")
                mf.write(f"Accuracy\t{acc:.4f}\n")
                mf.write(f"F1-Score\t{f1:.4f}\n")
                mf.write(f"Precision\t{precision:.4f}\n")
                mf.write(f"Recall\t{recall:.4f}\n")
            print(f"Dev metrics saved to {metrics_file}")
        
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * self.hparams.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

# ---------------------------------------------------------------------------
# 4. 메인 실행 스크립트
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Paraphrase Detection Training with MTAN methodology")
    
    # 데이터 경로
    parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
    
    # 출력 경로
    parser.add_argument("--para_dev_out", type=str, default="../predictions/para-dev-output-mtan.csv")
    parser.add_argument("--para_test_out", type=str, default="../predictions/para-test-output-mtan.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs")

    # 모델 및 학습 하이퍼파라미터
    parser.add_argument("--model_size", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length for tokenizer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    
    # 시스템 설정
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["16-mixed", "32-true"])
    parser.add_argument("--gpu_id", type=int, default=0, help="Index of the single GPU to use (set -1 for CPU)")

    return parser.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)

    output_path = Path(args.output_dir)
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model_size}_MTAN"

    datamodule = ParaphraseDataModule(hparams=args)
    model = ParaphraseMTANLitModule(hparams=args)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints" / run_name,
        filename="best-{epoch}-{val/acc:.4f}",
        save_top_k=3,
        monitor="val/acc",
        mode="max",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    tb_logger = TensorBoardLogger(output_path / "logs", name=run_name)
    txt_logger = TextFileLogger(output_path / "logs" / run_name / "train_log.txt")

    # ----- Lightning accelerator & devices 설정 -----
    if torch.cuda.is_available() and args.gpu_id >= 0:
        # 선택한 GPU로 디바이스 설정
        torch.cuda.set_device(args.gpu_id)
        accelerator_type = "gpu"
        lightning_devices = [args.gpu_id]    # Lightning에 실제 GPU 인덱스 전달
    else:
        accelerator_type = "cpu"
        lightning_devices = 1                 # CPU 1개 디바이스로 지정

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator_type,
        devices=lightning_devices,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor, txt_logger],
        logger=tb_logger,
        log_every_n_steps=50,
        deterministic=True
    )
    
    print("--- Starting Training ---")
    trainer.fit(model, datamodule)
    
    print("\n--- Starting Testing ---")
    # `ckpt_path='best'`를 사용하여 검증 성능이 가장 좋았던 체크포인트로 테스트
    trainer.test(model, datamodule=datamodule, ckpt_path='best')
    
    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    main()