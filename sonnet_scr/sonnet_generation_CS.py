'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation_CS.py --use_gpu`

학습 없이 기본 GPT2모델로 추론
Contrastive Search이용 구현
실행 후 추론결과와 정답데이터와 비교하여 자동으로 CHRF값 출력

입력 데이터: data/sonnets_held_out_dev.txt
정답 데이터: data/TRUE_sonnets_held_out_dev.txt
추론 결과: predictions/generated_sonnets_dev_cs.txt

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
    
  def get_device(self):
    for param in self.gpt.parameters():
      return param.device
    
  @torch.no_grad()
  def generate_contrastive_search(
      self,
      encoding,
      top_k: int = 8,
      penalty_alpha: float = 0.6,
      max_length: int = 128,
      no_repeat_ngram_size: int = 3,
      dynamic_alpha: bool = True,
      use_mean_sim: bool = True,
      use_repetition_penalty: bool = False,
      repetition_penalty: float = 1.1,
  ):
      """
      강화된 Contrastive Search 구현:
      - top_k: 후보 토큰 수
      - penalty_alpha: repulsion 강도
      - no_repeat_ngram_size: 반복 방지를 위한 n-gram 크기
      - dynamic_alpha: 단계별 α 조정 여부
      - use_mean_sim: 평균(sim.mean) vs 최대(sim.max) repulsion 점수 선택
      - use_repetition_penalty: 기존 토큰 반복 억제 여부
      - repetition_penalty: 반복 패널티 계수 (>1.0)
      """
      device = self.get_device()
      input_ids = encoding.to(device)
      attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

      rep_states = []
      max_history = 5
      for step in range(max_length):
          # 1) 로짓 및 히든 상태
          logits, hidden_states = self(
              input_ids=input_ids,
              attention_mask=attention_mask,
              output_hidden_states=True
          )
          last_hidden = hidden_states[:, -1, :]
          rep_states.append(last_hidden)
          if len(rep_states) > max_history:
              rep_states.pop(0)

          # 2) 다음 토큰 로짓
          next_logits = logits[:, -1, :].clone()

          # 3) 반복 패널티
          if use_repetition_penalty:
              for prev in input_ids[0].tolist():
                  next_logits[0, prev] /= repetition_penalty

          # 4) top-k 후보 추출
          topk_logits, topk_indices = torch.topk(next_logits, top_k, dim=-1)

          # 5) n-gram 반복 금지
          def ban_repeated_ngrams(seq: torch.Tensor, n: int):
              bans = set()
              s = seq.tolist()
              if len(s) >= n - 1:
                  prev = tuple(s[-(n - 1):])
                  for i in range(len(s) - n + 1):
                      if tuple(s[i:i + n - 1]) == prev:
                          bans.add(s[i + n - 1])
              return bans

          bans = ban_repeated_ngrams(input_ids[0], no_repeat_ngram_size)
          for i, token in enumerate(topk_indices[0]):
              if token.item() in bans:
                  topk_logits[0, i] = -float('Inf')

          # 6) 후보 배치 생성
          cand_input = input_ids.expand(top_k, -1).clone()
          cand_mask = attention_mask.expand(top_k, -1).clone()
          cand_input = torch.cat([cand_input, topk_indices.squeeze(0).unsqueeze(1)], dim=1)
          cand_mask = torch.cat([cand_mask, torch.ones((top_k, 1), dtype=torch.int64, device=device)], dim=1)

          # 7) 후보별 히든 상태
          _, cand_hidden_states = self(
              input_ids=cand_input,
              attention_mask=cand_mask,
              output_hidden_states=True
          )
          cand_hidden = cand_hidden_states[:, -1, :]

          # 8) repulsion 점수 (cosine similarity)
          rep_tensor = torch.cat(rep_states, dim=0)
          sims = torch.nn.functional.cosine_similarity(
              cand_hidden.unsqueeze(1), rep_tensor.unsqueeze(0), dim=-1
          )
          rep_scores = sims.mean(dim=1) if use_mean_sim else sims.max(dim=1)[0]

          # 9) α 동적 조정
          alpha = penalty_alpha * ((step + 1) / max_length) if dynamic_alpha else penalty_alpha

          # 10) 최종 점수 및 토큰 선택
          scores = topk_logits.squeeze(0) - alpha * rep_scores
          best = torch.argmax(scores)
          next_token = topk_indices[0, best].view(1, 1)

          # 11) 토큰 추가
          input_ids = torch.cat([input_ids, next_token], dim=1)
          attention_mask = torch.cat([
              attention_mask,
              torch.ones((1, 1), dtype=torch.int64, device=device)
          ], dim=1)

          if next_token.item() == self.tokenizer.eos_token_id:
              break

      # 디코딩
      generated = self.tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True)
      return input_ids, generated
  
@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

  args = add_arguments(args)
  model = SonnetGPT(args)  # ✅ checkpoint 불러오지 않고 새로 만듦
  model = model.to(device)
  model.eval()

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    
    # generate_contrastive_search

    prefix = batch[1]
    # 수정: 먼저 BatchEncoding에서 input_ids만 꺼내서 device로 이동
    encodings = model.tokenizer(
        prefix,
        return_tensors='pt',
        padding=False,
        truncation=True
    )
    input_ids = encodings['input_ids'].to(device)   # Tensor만 to(device)
    
    # Contrastive Search를 고정 파라미터로 호출
    token_ids, generated_text = model.generate_contrastive_search(
        input_ids,       # 입력된 토큰 텐서
        top_k=4,        # 후보 토큰 수
        penalty_alpha=0.6,  # repulsion 강도
        max_length=128      # 최대 길이
    )
    decoded_output = generated_text
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
  

  parser.add_argument("--gold_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")


  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dev_cs.txt") #저장 폴더 이름

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

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
    chrf_score = test_sonnet(test_path=args.sonnet_out, gold_path=args.gold_path)
    print(f"CHRF score: {chrf_score:.4f}")

if __name__ == "__main__":
  args = get_args()

  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  generate_submission_sonnets(args)
  generate_sonnets_from_checkpoint(args)