# ETPC dataset으로 멀티태스크 학습 후 QQP dataset으로 재학습
# Paper: EMNLP 2023 Paraphrase Types for Generation and Detection

# main_binary_transfer_pipeline.py
import argparse
import csv
import datetime
import os
import random
import logging
from pathlib import Path
import xml.etree.ElementTree as ET # XML 파서 임포트
import re  # XML 정규식 클렌징을 위한 모듈

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

# 메모리 문제 발생 시 주석 처리 가능
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
    """파일 및 콘솔에 동시 출력되는 로거 설정"""
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = log_dir / f'paraphrase_pipeline_{current_time}.log'

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
# 2. 데이터셋 클래스 (⭐️ ETPC용 XML 로더로 수정)
# =================================================================================

def load_etpc_data_from_xml(xml_filename: str):
    """ETPC.xml 파일에서 데이터를 로드합니다. XML이 손상된 경우 자동 복구를 시도합니다."""
    # XML 파싱 전용 내부 함수 (복구 로직 포함)
    def _safe_parse(path: str):
        try:
            return ET.parse(path)
        except ET.ParseError:
            # 잘못된 토큰이나 엔티티로 인해 파싱이 실패한 경우, 간단한 정규식 클렌징 후 재시도합니다.
            with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
                xml_text = fp.read()
            # 1) 허용되지 않는 제어 문자 제거
            xml_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', xml_text)
            # 2) 잘못 이스케이프된 & 기호를 &amp; 로 치환 (예: "Bread & Butter" → "Bread &amp; Butter")
            xml_text = re.sub(r'&(?!#?\w+;)', '&amp;', xml_text)
            # 3) 다시 파싱 시도
            return ET.ElementTree(ET.fromstring(xml_text))

    data = []
    tree = _safe_parse(xml_filename)
    root = tree.getroot()
    for pair in root.findall('text_pair'):
        try:
            sent1 = pair.find('sent1_raw').text
            sent2 = pair.find('sent2_raw').text
            # etpc_label을 이진 레이블로 사용합니다.
            label = int(pair.find('etpc_label').text)
            data.append({
                'sentence1': preprocess_string(sent1),
                'sentence2': preprocess_string(sent2),
                'is_duplicate': label
            })
        except (AttributeError, ValueError):
            # 필수 태그가 없거나 값이 잘못된 경우 건너뛰기
            pass
    print(f"Loaded {len(data)} examples from {xml_filename}")
    return data

def load_qqp_data(paraphrase_filename: str, split: str = 'train'):
    """기존의 QQP 데이터 로더 (이름 변경)"""
    data = []
    # ... (제공된 코드와 동일)
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

class BinaryParaphraseDataset(Dataset):
    """ETPC와 QQP 모두를 위한 이진 분류 데이터셋"""
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
        label = item.get('is_duplicate', -1) # test셋에는 label이 없음
        sent_id = item.get('id', idx) # ETPC에는 id가 없으므로 인덱스 사용
        
        prompt = f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '
        encoding = self.tokenizer(
            prompt, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'token_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'sent_ids': str(sent_id)
        }

class TestDataset(Dataset):
    """QQP 테스트 데이터셋 (id만 있음)"""
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
# 3. PyTorch Lightning 데이터 모듈 (⭐️ ETPC/QQP용으로 분리)
# =================================================================================

class ETPCDataModule(pl.LightningDataModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage: str = None):
        # ETPC.xml을 train/dev로 분할하여 사용 (예: 80/20)
        full_dataset = load_etpc_data_from_xml(self.hparams.etpc_data_path)
        train_size = int(0.8 * len(full_dataset))
        dev_size = len(full_dataset) - train_size
        train_raw, dev_raw = torch.utils.data.random_split(full_dataset, [train_size, dev_size])
        
        self.train_set = BinaryParaphraseDataset(list(train_raw), self.tokenizer, self.hparams.max_length)
        self.val_set = BinaryParaphraseDataset(list(dev_raw), self.tokenizer, self.hparams.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

class QQPDataModule(pl.LightningDataModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage: str = None):
        if stage in ("fit", None):
            train_raw = load_qqp_data(self.hparams.para_train, split='train')
            dev_raw = load_qqp_data(self.hparams.para_dev, split='dev')
            self.train_set = BinaryParaphraseDataset(train_raw, self.tokenizer, self.hparams.max_length)
            self.val_set = BinaryParaphraseDataset(dev_raw, self.tokenizer, self.hparams.max_length)
        if stage in ("test", None):
            dev_raw = load_qqp_data(self.hparams.para_dev, split='dev')
            test_raw = load_qqp_data(self.hparams.para_test, split='test')
            self.dev_for_test_set = BinaryParaphraseDataset(dev_raw, self.tokenizer, self.hparams.max_length)
            self.test_set = TestDataset(test_raw, self.tokenizer, self.hparams.max_length)

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
# 4. PyTorch Lightning 모델 (⭐️ 이진 분류로 단순화)
# =================================================================================

class ParaphraseLitModule(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        # hparams를 저장하여 체크포인트에서 로드/저장할 수 있도록 함
        self.save_hyperparameters(hparams)
        
        self.gpt = HFGPT2Model.from_pretrained(self.hparams.model_name_or_path)
        self.classification_head = nn.Linear(self.gpt.config.hidden_size, 2)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
            
        self.test_outputs = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = gpt_out.last_hidden_state

        if self.hparams.pooling_method == 'last':
            last_idx = attention_mask.sum(dim=1) - 1
            pooled_output = hidden_states[torch.arange(hidden_states.size(0)), last_idx]
        elif self.hparams.pooling_method == 'mean':
            expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden_states = torch.sum(hidden_states * expanded_mask, dim=1)
            sum_mask = torch.clamp(torch.sum(expanded_mask, dim=1), min=1e-9)
            pooled_output = sum_hidden_states / sum_mask
        elif self.hparams.pooling_method == 'cls':
            pooled_output = hidden_states[:, 0, :]
        else:
            raise ValueError("Unsupported pooling method. Choose from 'last', 'mean', 'cls'.")

        logits = self.classification_head(pooled_output)
        return logits

    def step(self, batch):
        logits = self(batch['token_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'])
        preds = torch.argmax(logits, dim=1)
        return loss, preds

    def training_step(self, batch, batch_idx):
        loss, preds = self.step(batch)
        self.train_acc(preds, batch['labels'])
        batch_size = batch['labels'].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self.step(batch)
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
        # ... (제공된 코드와 동일)
        if not self.test_outputs: return
        output_dir = Path(self.hparams.output_dir) / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        # 실행 이름에 task 정보를 포함하여 파일명이 겹치지 않게 함
        run_name = self.hparams.get("run_name", "run") 
        dev_out_path = output_dir / f"{run_name}-dev-output.csv"
        test_out_path = output_dir / f"{run_name}-test-output.csv"
        
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
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)

# =================================================================================
# 5. 메인 실행 스크립트 (⭐️ 파이프라인 로직 수정)
# =================================================================================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary-Transfer Paraphrase Detection Pipeline")
    
    # --- 경로 관련 ---
    parser.add_argument("--etpc_data_path", type=str, default="../data/ETPC.xml", help="Path to the full ETPC.xml data file.")
    parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
    parser.add_argument("--output_dir", type=str, default="../outputs_multitask2")

    # --- 모델 및 학습 하이퍼파라미터 ---
    parser.add_argument("--model_size", type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument("--pooling_method", type=str, default='last', choices=['last', 'mean', 'cls'])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs_etpc", type=int, default=10, help="Epochs for ETPC fine-tuning.")
    parser.add_argument("--epochs_qqp", type=int, default=5, help="Epochs for QQP fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-5)
    
    # --- 파이프라인 제어 ---
    parser.add_argument("--skip_etpc_training", action='store_true', help="Skip ETPC training and load a checkpoint directly.")
    parser.add_argument("--etpc_checkpoint_path", type=str, default=None, help="Path to a pre-trained ETPC checkpoint.")

    # --- 시스템 관련 ---
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
    run_name_base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model_size}_{args.pooling_method}"
    
    logger = setup_logging(output_path / "logs_txt")

    # --- STAGE 1: ETPC Fine-tuning ---
    etpc_best_checkpoint = args.etpc_checkpoint_path

    if not args.skip_etpc_training:
        logger.info("\n" + "="*50 + "\n S T A G E 1 : ETPC Fine-tuning \n" + "="*50)
        
        etpc_run_name = run_name_base + "_ETPC"
        hparams_etpc = argparse.Namespace(**vars(args), task='etpc', run_name=etpc_run_name, model_name_or_path=args.model_size)
        
        etpc_dm = ETPCDataModule(hparams_etpc)
        etpc_model = ParaphraseLitModule(hparams=hparams_etpc)
        
        etpc_checkpoint_callback = ModelCheckpoint(
            dirpath=output_path / "checkpoints" / etpc_run_name,
            filename="best-etpc-{epoch}-{val_acc:.4f}",
            save_top_k=3,
            monitor="val/acc",
            mode="max",
        )
        
        etpc_trainer = pl.Trainer(
            max_epochs=args.epochs_etpc,
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision if args.accelerator == 'gpu' else 32,
            callbacks=[etpc_checkpoint_callback],
            logger=TensorBoardLogger(save_dir=output_path / "logs", name=etpc_run_name),
            log_every_n_steps=50
        )
        
        etpc_trainer.fit(etpc_model, etpc_dm)
        etpc_best_checkpoint = etpc_checkpoint_callback.best_model_path
        logger.info(f"\nETPC fine-tuning finished. Best checkpoint saved at: {etpc_best_checkpoint}\n")

    # --- STAGE 2: QQP Fine-tuning ---
    if not etpc_best_checkpoint:
        logger.warning("ETPC training was skipped and no checkpoint was provided. Starting QQP training from scratch.")
        start_model_path = args.model_size
    else:
        logger.info(f"Starting QQP training from ETPC checkpoint: {etpc_best_checkpoint}")
        start_model_path = etpc_best_checkpoint

    logger.info("\n" + "="*50 + "\n S T A G E 2 : QQP Fine-tuning & Evaluation \n" + "="*50)
    
    qqp_run_name = run_name_base + "_QQP"
    hparams_qqp = argparse.Namespace(**vars(args), task='qqp', run_name=qqp_run_name, model_name_or_path=start_model_path)
    
    qqp_dm = QQPDataModule(hparams_qqp)
    # ckpt 경로인지 여부에 따라 로딩 방식 분기
    if os.path.isfile(start_model_path) and start_model_path.endswith('.ckpt'):
        # PL 체크포인트에서 가중치를 불러와 초기화
        qqp_model = ParaphraseLitModule.load_from_checkpoint(start_model_path, hparams=hparams_qqp, strict=False)
    else:
        qqp_model = ParaphraseLitModule(hparams=hparams_qqp)

    qqp_checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints" / qqp_run_name,
        filename="best-qqp-{epoch}-{val_acc:.4f}",
        save_top_k=3,
        monitor="val/acc",
        mode="max",
        save_last=True
    )
    
    qqp_trainer = pl.Trainer(
        max_epochs=args.epochs_qqp,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision if args.accelerator == 'gpu' else 32,
        callbacks=[qqp_checkpoint_callback],
        logger=TensorBoardLogger(save_dir=output_path / "logs", name=qqp_run_name),
        log_every_n_steps=50
    )

    logger.info("\n--- QQP 훈련을 시작합니다 ---")
    qqp_trainer.fit(qqp_model, qqp_dm)

    logger.info("\n--- QQP 테스트를 시작합니다 ---")
    qqp_trainer.test(datamodule=qqp_dm, ckpt_path='best')
    
    logger.info("\n--- 모든 파이프라인이 종료되었습니다 ---")
    logger.info(f"결과는 다음 경로에서 확인하세요: {output_path.resolve()}")

if __name__ == "__main__":
    main()