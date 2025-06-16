'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation_fine.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
학습 구조는 기존 train() 함수를 개선하여 변형함.
학습중 CERF평가해 가장 높은 모델 3개의 체크포인트만 유지하고 나머지는 지운다.


모델 평가를 원하면 main함수의
  # generate_submission_sonnets(args)
  # generate_sonnets_from_checkpoint(args)
  부분 주석을 해제하고
  train(args)를 주석처리하자
그리고 평가하고 싶은 모델(체크포인트)를 다음 위치에 지정하자
parser.add_argument("--checkpoint", type=str,default='./74_200-1e-05-sonnet.pt', help='불러올 모델 체크포인트 파일 경로')
  

GPU가 여러장이면 사용할 gpu번호를 아래에 적어주고 없으면 다음 코드를 지우면 된다.
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW
from sacrebleu.metrics import CHRF

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

TQDM_DISABLE = False


# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):

  #'''
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""
  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # 기본적으로, 전체 모델을 fine-tuning한다. TODO: 이것은 좋은 생각이 아닌 것 같다.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask, output_hidden_states: bool = False):
    """
    ParaphraseGPT의 forward pass와 유사하지만, 여기서는 시퀀스의 마지막 토큰뿐만 아니라 시퀀스의 각 토큰에 대한 logit을 생성하려고 한다.
    이를 통해, 마지막 토큰에 대한 다음 토큰의 분포만 학습하는 것이 아니라, 모델은 소네트를 구성하는 자연어 분포를 학습할 수 있다.
    """
    ##-----새로 작성한 코드-----
    # GPT-2 적용, 모든 token hidden seq 사용.
    gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = gpt_out['last_hidden_state']  # [batch, seq_len, hidden_dim]

    # weight tying. hidden state -> vocab_size의 logit output.
    logits = self.gpt.hidden_state_to_token(hidden_states)  # [batch, seq_len, vocab_size]

    if output_hidden_states:
      # (로짓, 히든스테이트) 튜플로 반환
      return logits, hidden_states

    return logits
    ##------------------------
  #'''

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device
  

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    top-p sampling 과 softmax temperature를 사용하여 새로운 소넷을 생성한다.

    TODO: 지금 이 방법은 기대 이하일 수 있다. 영감을 얻기 위해 Hugging Face의 model.generate(...) 함수를 참고해도 좋겠다.
        여러 시퀀스를 생성하고 beam search를 통해 최적의 시퀀스를 선택하는 것도 좋은 한 가지 방법이다.
        Top-k 샘플링 역시 또 다른 방법이며, 그 외에도 많은 접근법이 있다.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # logits을 구하기 위한 forward pass.
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    # prefix_len 만큼의 “가상 토큰”은 앞에서 잘라 내기
    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output
  
def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

from transformers import get_linear_schedule_with_warmup
def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(
      sonnet_dataset,
      shuffle=True,
      batch_size=args.batch_size,
      collate_fn=sonnet_dataset.collate_fn
  )

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)
  total_steps = len(sonnet_dataloader) * args.epochs
  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=int(0.1 * total_steps),
      num_training_steps=total_steps
  )
  # CHRF 기록 리스트
  model_scores = []
  deleted_paths = set()
  for epoch in range(args.epochs):
      model.train()
      train_loss = 0
      num_batches = 0

      for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
          b_ids, b_mask = batch['token_ids'], batch['attention_mask']
          b_ids = b_ids.to(device)
          b_mask = b_mask.to(device)

          optimizer.zero_grad()

          logits = model(b_ids, b_mask)
          shift_logits = logits[:, :-1].contiguous()  # [B, L-1, V]
          labels = b_ids[:, 1:].contiguous()          # [B, L-1]

          shift_logits = shift_logits.view(-1, shift_logits.size(-1))
          labels = labels.view(-1)
          loss = F.cross_entropy(shift_logits, labels, ignore_index=model.tokenizer.pad_token_id)

          loss.backward()

          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 필수!
          optimizer.step()
          scheduler.step()  # 추가!

          train_loss += loss.item()
          num_batches += 1

      avg_loss = train_loss / num_batches
      print(f"Epoch {epoch} - train loss: {avg_loss:.4f}")

      # Dev 평가 
      model.eval()
      chrf_metric = CHRF()

      generated_sonnets = []
      for sid, prefix in held_out_sonnet_dataset:
          encoding = model.tokenizer(
              prefix, return_tensors='pt', padding=False, truncation=True
          ).to(device)
          out_ids, _ = model.generate(
              encoding['input_ids'],
              temperature=args.temperature,
              top_p=args.top_p,
          )
          text = model.tokenizer.decode(
              out_ids[0].cpu().numpy().tolist(),
              skip_special_tokens=True
          )
          generated_sonnets.append(text)

      true_sonnets = [x[1] for x in SonnetsDataset(args.gold_path)]

      n = min(len(generated_sonnets), len(true_sonnets))
      if n > 0:
          dev_chrf = chrf_metric.corpus_score(generated_sonnets[:n], [true_sonnets[:n]]).score
          print(f"Epoch {epoch} - DEV CHRF: {dev_chrf:.2f}")

      model.train()

      # 모델 저장
      model_path = f"{epoch}_{args.filepath}"
      save_model(model, optimizer, args, model_path)
      model_scores.append((dev_chrf, model_path))

      # 최종 정리: 상위 3개만 남기고 나머지 삭제
      model_scores.sort(reverse=True, key=lambda x: x[0])  # CHRF 기준 내림차순 정렬
      top3 = set(score[1] for score in model_scores[:3])
      for _, path in model_scores:
        if path not in top3 and path not in deleted_paths:
            try:
                os.remove(path)
                deleted_paths.add(path)
                print(f"> Deleted model: {path}")
            except Exception as e:
                print(f"> Failed to delete {path}: {e}")


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  # saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)
  saved = torch.load(args.checkpoint, weights_only=False) #변경

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    
    # 기본
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)

    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

def get_args():
  parser = argparse.ArgumentParser()
  
  #----------------------- 추가함
  parser.add_argument("--gold_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--checkpoint", type=str,default='./74_200-1e-05-sonnet.pt', help='불러올 모델 체크포인트 파일 경로')
  #----------------------- 여기까지


  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dev_fine.txt") #저장 폴더 이름

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=200)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2) #1.2
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5) #1e-5
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
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

from evaluation import test_sonnet
def generate_sonnets_from_checkpoint(args):
    device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    seed_everything(11711)

    # 2) CHRF 평가 (evaluation.test_sonnet 재사용)
    chrf_score = test_sonnet(test_path=args.sonnet_out, gold_path=args.gold_path)
    print(f"CHRF score: {chrf_score:.4f}")

if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  # generate_submission_sonnets(args)
  # generate_sonnets_from_checkpoint(args)
