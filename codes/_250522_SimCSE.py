#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paraphrase detection with GPT-2 fine-tuning + SimCSE contrastive pre-training.
Mixed precision enabled via torch.cuda.amp

실행:
  python paraphrase_detection.py --use_gpu
"""

import argparse
import os
import random
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from transformers import GPT2Tokenizer
from torch.amp import autocast, GradScaler
import logging

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from optimizer import AdamW

TQDM_DISABLE = False


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


# GPU 사용 가능 여부와 개수 확인
def check_gpu():
  logger = logging.getLogger(__name__)
  if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    logger.info(f"GPU 사용 가능, 총 {n_gpu}개의 GPU가 감지되었습니다.")
    for i in range(n_gpu):
      gpu_name = torch.cuda.get_device_name(i)
      logger.info(f"  GPU {i}: {gpu_name}")
    return True, n_gpu
  else:
    logger.info("GPU를 사용할 수 없습니다. CPU를 사용합니다.")
    return False, 0


def create_directories():
  os.makedirs('checkpoints', exist_ok=True)
  os.makedirs('logs', exist_ok=True)
  os.makedirs('predictions', exist_ok=True)

def setup_logging():
  """로깅 설정"""
  current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
  log_file = os.path.join('logs', f'training_{current_time}.log')
  
  # 로깅 포맷 설정
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
      logging.FileHandler(log_file, encoding='utf-8'),
      logging.StreamHandler()  # 콘솔 출력도 유지
    ]
  )
  
  logger = logging.getLogger(__name__)
  logger.info(f"로깅이 시작되었습니다. 로그 파일: {log_file}")
  return logger

########################################
# SimCSE modules
########################################
class SimCSELoss(nn.Module):
  def __init__(self, temperature: float = 0.05):
    super().__init__()
    self.temperature = temperature

  def forward(self, emb: torch.Tensor) -> torch.Tensor:
    emb = F.normalize(emb, p=2, dim=1)
    sim = torch.matmul(emb, emb.T) / self.temperature
    eye = torch.eye(sim.size(0), dtype=torch.bool, device=emb.device)
    
    # float16에서 안전한 값 사용 (-1e4는 float16 범위 내)
    mask_value = -1e4 if sim.dtype == torch.float16 else -1e9
    sim.masked_fill_(eye, mask_value)
    
    # SimCSE positive pairs: 첫 번째 절반과 두 번째 절반이 서로의 positive
    batch_size = emb.size(0) // 2
    # 전체 배치에 대한 positive 인덱스 생성
    # 첫 번째 절반 [0,1,2,...,batch_size-1]의 positive는 [batch_size, batch_size+1, ..., 2*batch_size-1]
    # 두 번째 절반 [batch_size, batch_size+1, ..., 2*batch_size-1]의 positive는 [0,1,2,...,batch_size-1]
    pos = torch.cat([
        torch.arange(batch_size, 2*batch_size, device=emb.device),  # 첫 번째 절반의 positive
        torch.arange(0, batch_size, device=emb.device)               # 두 번째 절반의 positive
    ], dim=0)
    
    loss = F.cross_entropy(sim, pos)
    return loss

class SimCSEDataset(torch.utils.data.Dataset):
  def __init__(self, sent_list, tokenizer, max_len):
    self.sent_list = sent_list
    self.tok = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.sent_list)

  def __getitem__(self, idx):
    s = self.sent_list[idx]
    tokens = self.tok(
      s,
      truncation=True,
      max_length=self.max_len,
      padding='max_length',
      return_tensors='pt'
    )
    return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)


def simcse_collate(batch):
  ids, masks = zip(*batch)
  ids = torch.stack(ids)
  masks = torch.stack(masks)
  ids = torch.cat([ids, ids], 0)
  masks = torch.cat([masks, masks], 0)
  return ids, masks


def simcse_pretrain(args, model, tokenizer, device, use_gpu, n_gpu):
  logger = logging.getLogger(__name__)
  logger.info("=== SimCSE unsupervised pre-training 시작 ===")
  
  sents = []
  with open(args.para_train, encoding='utf8') as f:
    next(f)
    for line in f:
      line = line.rstrip('\n\r')
      if not line.strip(): continue
      parts = line.split('\t')
      if len(parts) >= 4:
        _, s1, s2, *_ = parts
        if s1.strip() and s2.strip():
          sents.extend([s1.strip(), s2.strip()])
      else:
        logger.warning(f"건너뛴 라인 (형식 불일치): {line[:50]}...")
  
  logger.info(f"SimCSE 훈련용 문장 수: {len(sents)}")
  
  # 멀티 GPU 설정
  if use_gpu and args.use_gpu and n_gpu > 1:
    logger.info(f"SimCSE 훈련에서 {n_gpu}개의 GPU를 병렬로 사용합니다.")
    model = torch.nn.DataParallel(model)
  
  dataset = SimCSEDataset(sents, tokenizer, args.max_len)
  loader = DataLoader(dataset,
                      batch_size=args.simcse_batch,
                      shuffle=True,
                      collate_fn=simcse_collate)
  criterion = SimCSELoss(args.simcse_tau).to(device)
  optimizer = AdamW(model.parameters(), lr=args.simcse_lr, weight_decay=0.)
  
  # CUDA가 사용 가능할 때만 mixed precision 사용
  if use_gpu and args.use_gpu:
    scaler = GradScaler('cuda')
  else:
    scaler = None
  
  model.train()

  for epoch in range(args.simcse_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for step, (ids, masks) in enumerate(
        tqdm(loader, desc=f'simcse-{epoch}', disable=TQDM_DISABLE)):
      ids, masks = ids.to(device), masks.to(device)
      optimizer.zero_grad()
      
      # Mixed precision 사용 여부에 따라 분기
      if scaler is not None:
        with autocast('cuda'):
          # DataParallel인 경우 module 속성으로 접근
          gpt_model = model.module.gpt if isinstance(model, torch.nn.DataParallel) else model.gpt
          h = gpt_model(input_ids=ids, attention_mask=masks)['last_hidden_state']
          cls = h[:, -1, :]
          loss = criterion(cls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        # CPU에서는 mixed precision 없이 실행
        gpt_model = model.module.gpt if isinstance(model, torch.nn.DataParallel) else model.gpt
        h = gpt_model(input_ids=ids, attention_mask=masks)['last_hidden_state']
        cls = h[:, -1, :]
        loss = criterion(cls)
        loss.backward()
        optimizer.step()
      
      epoch_loss += loss.item()
      num_batches += 1
      
      if (step + 1) % 500 == 0:
        logger.info(f'[SimCSE] epoch {epoch}, step {step+1}: loss {loss.item():.3f}')
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f'[SimCSE] Epoch {epoch} 완료 - 평균 손실: {avg_loss:.3f}')
    
  logger.info("=== SimCSE pre-training 종료 ===\n")
  
  return model

########################################
# Paraphrase detection modules
########################################
class ParaphraseGPT(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(
      args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
    )
    self.paraphrase_detection_head = nn.Linear(args.d, 2)
    for p in self.gpt.parameters():
      p.requires_grad = True

  def forward(self, input_ids, attention_mask):
    gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = gpt_out['last_hidden_state']
    last_idx = attention_mask.sum(dim=1) - 1
    last_token_state = hidden_states[
      torch.arange(hidden_states.size(0), device=hidden_states.device),
      last_idx
    ]
    logits = self.paraphrase_detection_head(last_token_state)
    return logits


def save_model(model, optimizer, args, filepath, is_best=False):
  if is_best:
    # DataParallel인 경우 module.state_dict() 사용
    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({
        'model': model_state,
        'optim': optimizer.state_dict(),
        'args': args
    }, os.path.join('checkpoints', 'best_model.pt'))


def train(args, pretrained_model=None):
  create_directories()
  logger = setup_logging()
  
  # TensorBoard 설정
  current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
  writer = SummaryWriter(log_dir=os.path.join('logs', f'tensorboard_{current_time}'))
  
  use_gpu, n_gpu = check_gpu()
  device = torch.device('cuda') if use_gpu and args.use_gpu else torch.device('cpu')
  
  logger.info(f"디바이스: {device}")
  logger.info(f"GPU 사용: {use_gpu and args.use_gpu}")
  if use_gpu and args.use_gpu:
    logger.info(f"GPU 개수: {n_gpu}")

  para_train = ParaphraseDetectionDataset(
    load_paraphrase_data(args.para_train), args)
  para_dev = ParaphraseDetectionDataset(
    load_paraphrase_data(args.para_dev), args)
  train_dl = DataLoader(
    para_train, shuffle=True, batch_size=args.batch_size,
    collate_fn=para_train.collate_fn)
  dev_dl = DataLoader(
    para_dev, shuffle=False, batch_size=args.batch_size,
    collate_fn=para_dev.collate_fn)

  logger.info(f"훈련 데이터 크기: {len(para_train)}")
  logger.info(f"검증 데이터 크기: {len(para_dev)}")
  logger.info(f"배치 크기: {args.batch_size}")
  logger.info(f"학습률: {args.lr}")
  logger.info(f"에포크 수: {args.epochs}")

  args = add_arguments(args)
  
  # 사전 훈련된 모델이 있으면 사용, 없으면 새로 생성
  if pretrained_model is not None:
    model = pretrained_model
    # DataParallel이 적용된 경우 원본 모델 추출
    if isinstance(model, torch.nn.DataParallel):
      model = model.module
  else:
    model = ParaphraseGPT(args).to(device)
  
  # 멀티 GPU 설정
  if use_gpu and args.use_gpu and n_gpu > 1:
    print(f"훈련에서 {n_gpu}개의 GPU를 병렬로 사용합니다.")
    model = torch.nn.DataParallel(model)
  
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
  
  # CUDA가 사용 가능할 때만 mixed precision 사용
  if use_gpu and args.use_gpu:
    scaler = GradScaler('cuda')
  else:
    scaler = None
    
  best_dev_acc = 0
  global_step = 0
  train_loss_accum = 0.0
  log_interval = 100  # 100 배치마다 로그 출력

  for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(train_dl, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].flatten().to(device)
      optimizer.zero_grad()
      
      # Mixed precision 사용 여부에 따라 분기
      if scaler is not None:
        with autocast('cuda'):
          logits = model(b_ids, b_mask)
          loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        # CPU에서는 mixed precision 없이 실행
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        
      # 손실 추적
      epoch_loss += loss.item()
      train_loss_accum += loss.item()
      num_batches += 1
      global_step += 1
      
      # 정기적으로 TensorBoard에 로그 기록
      if global_step % log_interval == 0:
        avg_loss = train_loss_accum / log_interval
        writer.add_scalar('Train/Loss', avg_loss, global_step)
        logger.info(f'Step {global_step}: 평균 훈련 손실 {avg_loss:.4f}')
        train_loss_accum = 0.0

    # 에포크 종료 후 평균 손실 계산
    avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f'Epoch {epoch} 훈련 완료 - 평균 손실: {avg_epoch_loss:.4f}')

    model.eval()
    with torch.no_grad():
      dev_acc, dev_f1, *_ = model_eval_paraphrase(dev_dl, model, device)
    
    # TensorBoard에 검증 메트릭 기록
    writer.add_scalar('Validation/Accuracy', dev_acc, epoch)
    writer.add_scalar('Validation/F1', dev_f1, epoch)
    writer.add_scalar('Train/EpochLoss', avg_epoch_loss, epoch)
    
    is_best = dev_acc > best_dev_acc
    if is_best:
      logger.info(f'새로운 최고 성능! 이전: {best_dev_acc:.3f} -> 현재: {dev_acc:.3f}')
    
    best_dev_acc = max(best_dev_acc, dev_acc)
    save_model(model, optimizer, args,
               os.path.join('checkpoints', f'epoch_{epoch}.pt'), is_best)
    
    logger.info(f'Epoch {epoch}: dev acc={dev_acc:.3f}, dev f1={dev_f1:.3f} (최고: {best_dev_acc:.3f})')
    model.train()

  logger.info(f"훈련 완료! 최종 최고 성능: {best_dev_acc:.3f}")
  writer.close()


@torch.no_grad()
def test(args):
  logger = logging.getLogger(__name__)
  logger.info("=== 테스트 시작 ===")
  
  use_gpu, n_gpu = check_gpu()
  device = torch.device('cuda') if use_gpu and args.use_gpu else torch.device('cpu')
  
  best_path = os.path.join('checkpoints', 'best_model.pt')
  if os.path.exists(best_path):
    logger.info(f"최고 성능 모델 로드: {best_path}")
    saved = torch.load(best_path)
  else:
    logger.info(f"기본 모델 로드: {args.filepath}")
    saved = torch.load(args.filepath)
    
  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  
  # 멀티 GPU 설정 (모델 로드 후에 설정)
  if use_gpu and args.use_gpu and n_gpu > 1:
    logger.info(f"테스트에서 {n_gpu}개의 GPU를 병렬로 사용합니다.")
    model = torch.nn.DataParallel(model)
  
  model.eval()

  para_dev = ParaphraseDetectionDataset(
    load_paraphrase_data(args.para_dev), args)
  para_test = ParaphraseDetectionTestDataset(
      load_paraphrase_data(args.para_test, split='test'), args)

  dev_dl = DataLoader(
    para_dev, shuffle=False, batch_size=args.batch_size,
    collate_fn=para_dev.collate_fn)
  test_dl = DataLoader(
    para_test, shuffle=False, batch_size=args.batch_size,
    collate_fn=para_test.collate_fn)

  logger.info("개발 세트 평가 중...")
  dev_acc, dev_f1, dev_pred, _, dev_ids = model_eval_paraphrase(
    dev_dl, model, device)
  logger.info(f'개발 세트 결과: acc={dev_acc:.3f}, f1={dev_f1:.3f}')
  
  logger.info("테스트 세트 예측 중...")
  test_pred, test_ids = model_test_paraphrase(test_dl, model, device)

  create_directories()
  logger.info(f"예측 결과 저장: {args.para_dev_out}, {args.para_test_out}")
  with open(args.para_dev_out, 'w') as f:
    f.write('id,Predicted_Is_Paraphrase\n')
    for pid, p in zip(dev_ids, dev_pred):
      f.write(f'{pid},{p}\n')
  with open(args.para_test_out, 'w') as f:
    f.write('id,Predicted_Is_Paraphrase\n')
    for pid, p in zip(test_ids, test_pred):
      f.write(f'{pid},{p}\n')
  
  logger.info("=== 테스트 완료 ===")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="../data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="../data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="../data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="../predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="../predictions/para-test-output.csv")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--model_size", type=str,
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  # SimCSE
  parser.add_argument("--simcse_epochs", type=int, default=1)
  parser.add_argument("--simcse_batch", type=int, default=64)
  parser.add_argument("--simcse_lr", type=float, default=5e-5)
  parser.add_argument("--simcse_tau", type=float, default=0.05)
  parser.add_argument("--max_len", type=int, default=128)

  args = parser.parse_args()
  return args


def add_arguments(args):
  if args.model_size == 'gpt2':
    args.d, args.l, args.num_heads = 768, 12, 12
  elif args.model_size == 'gpt2-medium':
    args.d, args.l, args.num_heads = 1024, 24, 16
  elif args.model_size == 'gpt2-large':
    args.d, args.l, args.num_heads = 1280, 36, 20
  else:
    raise ValueError(f'{args.model_size} unsupported.')
  return args

if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'
  seed_everything(args.seed)
  
  # 초기 로깅 설정
  logger = setup_logging()
  logger.info("=== 프로그램 시작 ===")
  logger.info(f"시드: {args.seed}")
  logger.info(f"설정 파라미터:")
  logger.info(f"  - 모델 크기: {args.model_size}")
  logger.info(f"  - 에포크: {args.epochs}")
  logger.info(f"  - 배치 크기: {args.batch_size}")
  logger.info(f"  - 학습률: {args.lr}")
  logger.info(f"  - SimCSE 에포크: {args.simcse_epochs}")
  logger.info(f"  - SimCSE 배치 크기: {args.simcse_batch}")
  logger.info(f"  - SimCSE 학습률: {args.simcse_lr}")
  logger.info(f"  - GPU 사용: {args.use_gpu}")
  
  use_gpu, n_gpu = check_gpu()
  device = torch.device('cuda') if use_gpu and args.use_gpu else torch.device('cpu')
  tmp_args = add_arguments(args)
  gpt_model = ParaphraseGPT(tmp_args).to(device)
  logger.info(f"모델 파라미터 수: {sum(p.numel() for p in gpt_model.parameters()):,}")

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer.pad_token = tokenizer.eos_token
  logger.info("토크나이저 로드 완료")
  
  # SimCSE pretraining
  logger.info("SimCSE 사전 훈련 시작")
  pretrained_model = simcse_pretrain(args, gpt_model, tokenizer, device, use_gpu, n_gpu)
  
  # Paraphrase detection training (사전 훈련된 모델 사용)
  logger.info("Paraphrase detection 훈련 시작")
  train(args, pretrained_model)
  
  # Test
  logger.info("테스트 시작")
  test(args)
  
  logger.info("=== 프로그램 완료 ===")
