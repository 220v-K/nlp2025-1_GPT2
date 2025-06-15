# paraphrase_detection_pl.py
# 전체 파이프라인: Scheduler + Mixed-Precision + Label-Smoothing + R-Drop + SimCSE
# gradient checkpointing, accumulate_grad_batches 적용
# PyTorch-Lightning 기반

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2Model,
    get_linear_schedule_with_warmup,
)
from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data,
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
import torchmetrics


# ---------------------------------------------------------------------------
# 0. 공통 유틸
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# 1. DataModule
# ---------------------------------------------------------------------------
class ParaphraseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        dev_path: str,
        test_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        model_size: str = "gpt2",
        max_length: int = 128
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_path", "dev_path", "test_path"])
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

    def setup(self, stage=None):
        if stage in ("fit", None):
            train_raw = load_paraphrase_data(self.train_path, split="train")
            dev_raw = load_paraphrase_data(self.dev_path, split="dev")
            self.train_set = ParaphraseDetectionDataset(
                train_raw, argparse.Namespace(model_size=self.hparams.model_size, max_length=self.hparams.max_length)
            )
            self.dev_set = ParaphraseDetectionDataset(
                dev_raw, argparse.Namespace(model_size=self.hparams.model_size, max_length=self.hparams.max_length)
            )

        if stage in ("test", None):
            test_raw = load_paraphrase_data(self.test_path, split="test")
            self.test_set = ParaphraseDetectionTestDataset(
                test_raw, argparse.Namespace(model_size=self.hparams.model_size, max_length=self.hparams.max_length)
            )

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
            self.dev_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dev_set.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.test_set.collate_fn,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# 2. LightningModule
# ---------------------------------------------------------------------------
class ParaphraseLightningModel(pl.LightningModule):
    def __init__(
        self,
        model_size: str = "gpt2",
        lr: float = 1e-5,
        label_smoothing: float = 0.1,
        rdrop_alpha: float = 5.0,
        simcse_temp: float = 0.05,
        warmup_ratio: float = 0.1,
        max_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        # --- 모델 ---
        self.gpt = GPT2Model.from_pretrained(model_size)
        # gradient checkpointing 켜기
        self.gpt.config.use_cache = False
        self.gpt.gradient_checkpointing_enable()
        #
        self.cls_head = nn.Linear(self.gpt.config.hidden_size, 2)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # metrics (step, epoch)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    # ---------------- util ----------------
    def _encode(self, input_ids, attention_mask):
        out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        last_pos = attention_mask.sum(dim=1) - 1  # 마지막 실제 토큰 위치
        rep = last_hidden[torch.arange(last_hidden.size(0), device=self.device), last_pos]
        logits = self.cls_head(rep)
        return logits, rep

    @staticmethod
    def _kl(p, q):
        return F.kl_div(F.log_softmax(p, -1), F.softmax(q, -1), reduction="batchmean")

    # ---------------- forward ----------------
    def forward(self, input_ids, attention_mask):
        return self._encode(input_ids, attention_mask)[0]

    # ---------------- training ----------------
    def training_step(self, batch, batch_idx):
        ids, mask, labels = batch["token_ids"], batch["attention_mask"], batch["labels"]
        # 두 번 forward (dropout 차이) -> R-Drop + SimCSE
        logits1, rep1 = self._encode(ids, mask)
        logits2, rep2 = self._encode(ids, mask)

        # CE + label smoothing
        ce1 = F.cross_entropy(logits1, labels, label_smoothing=self.hparams.label_smoothing)
        ce2 = F.cross_entropy(logits2, labels, label_smoothing=self.hparams.label_smoothing)
        ce_loss = 0.5 * (ce1 + ce2)

        # R-Drop KL
        kl_loss = 0.5 * (self._kl(logits1, logits2) + self._kl(logits2, logits1))

        # SimCSE (unsupervised) – 두 view를 positive 로 보고 InfoNCE
        batch_size = rep1.size(0)
        z = torch.cat([rep1, rep2], dim=0)  # [2B, d]
        z = F.normalize(z, dim=-1)
        sim = torch.matmul(z, z.T) / self.hparams.simcse_temp
        mask = torch.eye(2 * batch_size, device=self.device).bool()
        sim.masked_fill_(mask, -1e4)

        sim_labels = torch.arange(batch_size, device=self.device)
        sim_labels = torch.cat([sim_labels + batch_size, sim_labels], dim=0)
        sim_loss = F.cross_entropy(sim, sim_labels)

        total_loss = ce_loss + self.hparams.rdrop_alpha * kl_loss + sim_loss

        # logging
        self.log_dict(
            {
                "train_ce": ce_loss,
                "train_kl": kl_loss,
                "train_sim": sim_loss,
                "train_loss": total_loss,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=ids.size(0)
        )
        preds = torch.argmax(logits1, dim=1)
        self.train_acc(preds, labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=ids.size(0))
        return total_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 매 배치 끝날 때마다 캐시 비우기
        torch.cuda.empty_cache()

    # ---------------- validation ----------------
    def validation_step(self, batch, batch_idx):
        ids, mask, labels = batch["token_ids"], batch["attention_mask"], batch["labels"]
        logits, _ = self._encode(ids, mask)
        val_loss = F.cross_entropy(
            logits, labels, label_smoothing=self.hparams.label_smoothing
        )
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, labels)
        
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return val_loss

    def on_validation_epoch_end(self):
        pass

    # ---------------- optimizer / scheduler ----------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        
        # Calculate total_steps using the trainer's estimated_stepping_batches property directly
        # This property usually gives the total number of optimization steps over all epochs.
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "linear_warmup",
            },
        }


# ---------------------------------------------------------------------------
# 3. main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="../data/quora-train.csv")
    p.add_argument("--dev", default="../data/quora-dev.csv")
    p.add_argument("--test", default="../data/quora-test-student.csv")
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--model_size", choices=["gpt2", "gpt2-medium", "gpt2-large"], default="gpt2")
    p.add_argument("--precision", choices=["16-mixed", "32-true"], default="16-mixed")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=11711)
    p.add_argument("--max_length", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    # PYTORCH_CUDA_ALLOC_CONF 환경 변수 설정 (OOM 방지 및 메모리 단편화 완화)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 파이프라인 시작 전 GPU 캐시 비우기
    torch.cuda.empty_cache()

    # TensorCore 활용을 위한 설정 (PyTorch 1.12+ 권장)
    # AMP (Automatic Mixed Precision) 사용 시 matmul 연산 정밀도 설정
    if torch.cuda.is_available() and args.precision == "16-mixed": # bf16-mixed도 가능
        torch.set_float32_matmul_precision('medium') # 또는 'medium'

    # Data
    dm = ParaphraseDataModule(
        train_path=args.train,
        dev_path=args.dev,
        test_path=args.test,
        batch_size=args.batch_size,
        num_workers=4,
        model_size=args.model_size,
        max_length=args.max_length
    )

    # Model
    model = ParaphraseLightningModel(
        model_size=args.model_size,
        lr=args.lr,
        max_epochs=args.epochs,
    )

    # Callbacks
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-{epoch}-{val_acc:.4f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    logger = pl.loggers.TensorBoardLogger("logs", name="250523_simcse_etc_paraphrase_pl")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,
        precision=args.precision,
        accumulate_grad_batches=4,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, dm)

    # ---------- test ----------
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
