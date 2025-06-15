'''
Paraphrase detection을 위한 시작 코드.

고려 사항:
 - ParaphraseGPT: 여러분이 구현한 GPT-2 분류 모델 .
 - train: Quora paraphrase detection 데이터셋에서 ParaphraseGPT를 훈련시키는 절차.
 - test: Test 절차. 프로젝트 결과 제출에 필요한 파일들을 생성함.

실행:
  `python paraphrase_detection.py --use_gpu`
ParaphraseGPT model을 훈련 및 평가하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


# 필요한 디렉토리 생성
def create_directories():
  os.makedirs('checkpoints', exist_ok=True)
  os.makedirs('logs', exist_ok=True)
  os.makedirs('predictions', exist_ok=True)


class ParaphraseGPT(nn.Module):
  """Paraphrase Detection을 위해 설계된 여러분의 GPT-2 Model."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(
        args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
    )
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection 의 출력은 두 가지: 1 (yes) or 0 (no).

    # 기본적으로, 전체 모델을 finetuning 한다.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    TODO: paraphrase_detection_head Linear layer를 사용하여 토큰의 레이블을 예측하시오.

    입력은 다음과 같은 구조를 갖는다:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    따라서, 문장의 끝에서 다음 토큰에 대한 예측을 해야 할 것이다. 
    훈련이 잘 되었다면, 패러프레이즈인 경우에는 토큰 "yes"(BPE index 8505)가, 
    패러프레이즈가 아닌 경우에는 토큰 "no" (BPE index 3919)가 될 것이다.
    """
    ###-----추가한 코드----
    # GPT-2 실행
    gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)

    # 마지막 실제 토큰의 hidden state 획득
    if isinstance(gpt_out, tuple):
      hidden_states = gpt_out[0]
    elif isinstance(gpt_out, dict) and 'last_token' in gpt_out:
      return self.paraphrase_detection_head(gpt_out['last_token'])
    else:
      hidden_states = gpt_out.last_hidden_state

    # 마지막 토큰 인덱스 찾기
    last_idx = attention_mask.sum(dim=1) - 1                       # [batch]
    last_token_state = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device),
                                     last_idx]                     # [batch, hidden]

    # 분류 헤드
    logits = self.paraphrase_detection_head(last_token_state)        # [batch, 2]
    return logits
    ###--------------------



def save_model(model, optimizer, args, filepath, is_best=False):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"모델 저장 완료: {filepath}")

  # 최고 성능 모델일 경우 별도 복사
  if is_best:
    best_path = os.path.join('checkpoints', 'best_model.pt')
    torch.save(save_info, best_path)
    print(f"최고 성능 모델 저장 완료: {best_path}")


def train(args):
  """Quora 데이터셋에서 Paraphrase Detection을 위한 GPT-2 훈련."""
  # 필요한 디렉토리 생성
  create_directories()

  # 현재 시간을 로그 디렉토리 이름에 추가하여 실행마다 고유한 로그 생성
  current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
  log_dir = os.path.join('logs', f'run_{current_time}')
  
  # TensorBoard 설정
  writer = SummaryWriter(log_dir=log_dir)
  print(f"텐서보드 로그 저장 위치: {log_dir}")
  
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0
  
  # 체크포인트 경로 설정
  checkpoint_path = os.path.join('checkpoints', args.filepath)
  
  # 전역 스텝 및 로깅 설정
  global_step = 0
  log_interval = 100  # 로깅 간격(스텝 단위)
  
  # 메모리 모니터링용
  if args.use_gpu:
    peak_memory_allocated = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    num_batches = 0
    total_samples = 0
    
    batch_loss = 0
    batch_acc = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
      # 입력을 가져와서 GPU로 보내기
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      preds = torch.argmax(logits, dim=1)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      
      # 메모리 사용량 체크 및 캐시 클리어 (간헐적으로)
      if args.use_gpu and global_step % 10 == 0:
        current_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB 단위
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB 단위
        if peak_memory > peak_memory_allocated:
          peak_memory_allocated = peak_memory

      # 로깅을 위한 스텝 정보 추적
      global_step += 1
      
      # 학습 지표 추적
      with torch.no_grad():
        curr_acc = (preds == labels).float().mean().item()
        curr_loss = loss.item()
        
        batch_acc += curr_acc * len(labels)
        batch_loss += curr_loss * len(labels)
        batch_count += len(labels)
        
        train_acc += curr_acc * len(labels)
        train_loss += curr_loss * len(labels)
        
      total_samples += len(labels)
      num_batches += 1
      
      # 메모리에서 불필요한 텐서 제거
      del logits, preds, loss
      
      # 일정 간격으로 로깅
      if global_step % log_interval == 0 and batch_count > 0:
        avg_batch_loss = batch_loss / batch_count
        avg_batch_acc = batch_acc / batch_count
        
        # 성능 지표를 그룹화하여 시계열 그래프로 표시
        writer.add_scalars('Training', {
            'loss': avg_batch_loss,
            'accuracy': avg_batch_acc,
        }, global_step)
        
        # GPU 메모리 사용량을 별도의 그래프로 표시
        if args.use_gpu:
          writer.add_scalars('Memory_Usage', {
              'current_mb': current_memory,
              'peak_mb': peak_memory,
          }, global_step)
          
        print(f"Epoch {epoch}, Step {global_step}: train loss :: {avg_batch_loss:.3f}, train acc :: {avg_batch_acc:.3f}")
        
        # 로깅 후 배치 추적 변수 초기화
        batch_loss = 0
        batch_acc = 0
        batch_count = 0
        
        # VRAM 정리
        if args.use_gpu:
          torch.cuda.empty_cache()
      
    # 에폭 끝에 전체 성능 계산
    train_loss = train_loss / total_samples
    train_acc = train_acc / total_samples

    # 에폭 끝에 검증 성능 평가
    model.eval()
    with torch.no_grad():
      dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)
    model.train()

    # 에폭 단위 로깅 - 단일 그래프에 여러 지표 표시
    writer.add_scalars('Epoch_Training', {
        'loss': train_loss,
        'accuracy': train_acc,
    }, epoch)
    
    writer.add_scalars('Epoch_Validation', {
        'accuracy': dev_acc,
        'f1_score': dev_f1,
    }, epoch)
    
    # 에폭별 훈련/검증 정확도 비교
    writer.add_scalars('Epoch_Accuracy', {
        'train': train_acc,
        'validation': dev_acc,
    }, epoch)

    # 마지막 체크포인트 저장
    last_checkpoint = os.path.join('checkpoints', f'epoch_{epoch}.pt')
    save_model(model, optimizer, args, last_checkpoint)

    # 최고 성능 모델 저장
    is_best = dev_acc > best_dev_acc
    if is_best:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, checkpoint_path, is_best=True)

    print(f"Epoch {epoch} 완료: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")
    
    # 에폭 끝에 VRAM 정리
    if args.use_gpu:
      torch.cuda.empty_cache()

  # 학습 완료 후 마지막 모델 저장
  final_checkpoint = os.path.join('checkpoints', 'final_model.pt')
  save_model(model, optimizer, args, final_checkpoint)
  
  if args.use_gpu:
    print(f"훈련 중 최대 메모리 사용량: {peak_memory_allocated:.2f} MB")
  
  # 학습에 사용된 하이퍼파라미터 기록 (커스텀 텍스트 로깅)
  writer.add_text('Hyperparameters', 
                 f"모델 크기: {args.model_size}, 학습률: {args.lr}, 배치 크기: {args.batch_size}, 에폭: {args.epochs}", 0)
  
  writer.close()


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  # 최고 성능 모델 불러오기
  best_model_path = os.path.join('checkpoints', 'best_model.pt')
  if os.path.exists(best_model_path):
    saved = torch.load(best_model_path)
    print(f"최고 성능 모델을 불러옵니다: {best_model_path}")
  else:
    checkpoint_path = os.path.join('checkpoints', args.filepath)
    if os.path.exists(checkpoint_path):
      saved = torch.load(checkpoint_path)
      print(f"체크포인트를 불러옵니다: {checkpoint_path}")
    else:
      saved = torch.load(args.filepath)
      print(f"기존 모델을 불러옵니다: {args.filepath}")

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, dev_para_f1, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}, dev f1 :: {dev_para_f1 :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  # predictions 디렉토리 확인
  os.makedirs('predictions', exist_ok=True)
  
  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """모델 크기에 따라 결정되는 인수들을 추가."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  test(args)