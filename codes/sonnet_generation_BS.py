'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation_BS.py --use_gpu`

학습 없이 기본 GPT2모델로 추론
beam search 기법 추가
실행 후 추론결과와 정답데이터와 비교하여 자동으로 CHRF값 출력

입력 데이터: data/sonnets_held_out_dev.txt
정답 데이터: data/TRUE_sonnets_held_out_dev.txt
추론 결과: predictions/generated_sonnets_dev_bs.txt

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
  def generate_beam_search(self, encoding, beam_size=5, max_len=128, length_penalty=0.6, ngram_size=3):
      device = self.get_device()
      # beam: list of (token_ids, score)
      beams = [(encoding.to(device), 0.0)]
      completed = []
      for _ in range(max_len):
          new_beams = []
          for seq, score in beams:
              logits = self.forward(seq, torch.ones_like(seq))
              logp = F.log_softmax(logits[0, -1], dim=-1)  # [vocab]
              # n-gram block
              if ngram_size > 1:
                  banned = set()
                  last = seq[0].tolist()
                  if len(last) >= ngram_size-1:
                      prev = tuple(last[-(ngram_size-1):])
                      for i in range(len(last)-ngram_size+1):
                          if tuple(last[i:i+ngram_size-1]) == prev:
                              banned.add(last[i+ngram_size-1])
                  for tok in banned:
                      logp[tok] = -1e9
              # top-k for beam
              topk_logp, topk_idx = torch.topk(logp, beam_size)
              for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                  new_seq = torch.cat([seq, torch.tensor([[idx]], device=device)], dim=1)
                  new_score = score + lp
                  new_beams.append((new_seq, new_score))
          # length normalization & keep top beams
          beams = sorted(new_beams, key=lambda x: x[1] / ((x[0].size(1))**length_penalty), reverse=True)[:beam_size]
          # check eos
          beams, done = [], beams
          for seq, sc in done:
              if seq[0, -1].item() == self.tokenizer.eos_token_id:
                  completed.append((seq, sc))
              else:
                  beams.append((seq, sc))
          if not beams:
              break
      # pick best completed or best beam
      best_seq = max(completed, key=lambda x: x[1])[0] if completed else beams[0][0]
      text = self.tokenizer.decode(best_seq[0].cpu().tolist(), skip_special_tokens=True)
      return best_seq, text

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
    
    
    #beam
    encodings = model.tokenizer(
        batch[1],
        return_tensors='pt',
        padding=False,
        truncation=True
    )
    input_ids = encodings['input_ids'].to(device)

    # (2) Beam Search 호출
    #    beam_size, max_len, length_penalty, ngram_size는 하이퍼파라미터로 조정하세요.
    best_seq, decoded_output = model.generate_beam_search(
        input_ids,
        beam_size=5,         # 빔 크기
        max_len=128,         # 최대 길이
        length_penalty=0.6,  # 길이 보정 인자
        ngram_size=3         # n-그램 반복 차단 크기
    )
    
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
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dev_bs.txt") #저장 폴더 이름

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