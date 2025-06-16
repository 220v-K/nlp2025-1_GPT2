'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation_CE.py --use_gpu`

4.2.6. Candidate Ensemble + MBR
학습 없이 기본 GPT2모델로 추론
Top-p Sampling, Contrastive Search, Beam Search를 통해 생성된 다양한 후보군을 모두 통합하고,
이들 간의 pairwise CHRF 유사도를 기반으로 MBR reranking을 수행
실행 후 추론결과와 정답데이터와 비교하여 자동으로 CHRF값 출력

입력 데이터: data/sonnets_held_out_dev.txt
정답 데이터: data/TRUE_sonnets_held_out_dev.txt
추론 결과: predictions/generated_sonnets_dev_ce.txt

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

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output

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
  def _beam_search_all(
      self,
      input_ids,
      attention_mask=None,
      beam_size: int = 5,
      max_len: int = 128,
      length_penalty: float = 0.6,
      ngram_size: int = 3
  ):
      """
      Beam Search로 상위 nbest 후보 반환
      - input_ids: [1, prefix_len]
      - attention_mask: optional
      - beam_size: 빔 크기(=nbest)
      - max_len: 최대 생성 길이
      - length_penalty: 길이 정규화 계수
      - ngram_size: n-gram 반복 차단
      returns: List[(seq_tensor, score)] 정렬된 리스트
      """
      device = self.get_device()
      if attention_mask is None:
          attention_mask = torch.ones_like(input_ids, device=device)
      # beams: list of (seq, score)
      beams = [(input_ids, 0.0)]
      completed = []

      for _ in range(max_len):
          all_candidates = []
          for seq, score in beams:
              # forward
              logits = self.forward(seq, torch.ones_like(seq, device=device))
              logp = F.log_softmax(logits[0, -1], dim=-1)  # [vocab]
              # n-gram 차단
              if ngram_size > 1:
                  seq_list = seq[0].tolist()
                  if len(seq_list) >= ngram_size - 1:
                      prev = tuple(seq_list[-(ngram_size-1):])
                      for i in range(len(seq_list)-ngram_size+1):
                          if tuple(seq_list[i:i+ngram_size-1]) == prev:
                              logp[seq_list[i+ngram_size-1]] = -1e9
              # top-k 후보
              topk_logp, topk_idx = torch.topk(logp, beam_size)
              for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                  new_seq = torch.cat([seq, torch.tensor([[idx]], device=device)], dim=1)
                  new_score = score + lp
                  all_candidates.append((new_seq, new_score))
          # 길이 정규화후 상위 beam_size 유지
          beams = sorted(
              all_candidates,
              key=lambda x: x[1] / ((x[0].size(1)) ** length_penalty),
              reverse=True
          )[:beam_size]
          # EOS 분리
          next_beams = []
          for seq, score in beams:
              if seq[0, -1].item() == self.tokenizer.eos_token_id:
                  completed.append((seq, score))
              else:
                  next_beams.append((seq, score))
          beams = next_beams
          if not beams:
              break
      # 남은 beams도 completed에 추가
      completed.extend(beams)
      # 최종 정렬
      completed = sorted(
          completed,
          key=lambda x: x[1] / ((x[0].size(1)) ** length_penalty),
          reverse=True
      )
      return completed

  @torch.no_grad()
  def generate_mbr(
      self,
      input_ids,
      attention_mask=None,
      beam_size: int = 5,
      max_len: int = 128,
      length_penalty: float = 0.6,
      ngram_size: int = 3,
      nbest: int = 5
  ):
      """
      MBR Reranking:
      1) beam search로 상위 nbest 후보 생성
      2) 후보들 간 pairwise CHRF 유사도 계산
      3) 평균 유사도 최대 후보 선택
      """
      device = self.get_device()
      if attention_mask is None:
          attention_mask = torch.ones_like(input_ids, device=device)
      # 1) nbest 후보
      candidates = self._beam_search_all(
          input_ids,
          attention_mask,
          beam_size=beam_size,
          max_len=max_len,
          length_penalty=length_penalty,
          ngram_size=ngram_size
      )[:nbest]
      
      texts = []
      for seq, _ in candidates:
          # seq: Tensor of shape [1, seq_len], squeeze 후 1차원 리스트로 변환
          token_list = seq.squeeze(0).cpu().tolist()
          text = self.tokenizer.decode(token_list, skip_special_tokens=True)
          texts.append(text)

      # 2) pairwise CHRF
      metric = CHRF()
      scores = []
      for i, txt in enumerate(texts):
          others = texts[:i] + texts[i+1:]
          score = sum(metric.sentence_score(txt, [o]).score for o in others)
          scores.append(score)
      # 3) 대표성 최대 시퀀스
      best_idx = int(torch.tensor(scores).argmax())
      return candidates[best_idx][0], texts[best_idx]
  
  @torch.no_grad()
  def generate_all_candidates(self, encoding):
      input_ids = encoding.to(self.get_device())

      # 1. Top-p Sampling 후보들 (2~3개)
      top_p_candidates = [self.generate(input_ids, temperature=0.7, top_p=0.9)[1] for _ in range(3)]

      # 2. Contrastive 후보들 (2~3개)
      contrastive_candidates = [
          self.generate_contrastive_search(input_ids, top_k=4, penalty_alpha=0.6)[1]
          for _ in range(3)
      ]

      # 3. Beam search 후보들 (beam_size 크게 설정)
      beam_candidates = [
          self.generate_beam_search(input_ids, beam_size=10, length_penalty=0.8, ngram_size=4)[1]
          for _ in range(3)
      ]

      # 후보 전체 모음
      all_candidates = top_p_candidates + contrastive_candidates + beam_candidates
      return all_candidates
  
  @torch.no_grad()
  def generate_mbr_from_candidates(self, candidates):
    metric = CHRF()
    scores = []
    for i, cand in enumerate(candidates):
        others = candidates[:i] + candidates[i+1:]
        score = sum(metric.sentence_score(cand, [o]).score for o in others)
        scores.append(score)

    best_idx = int(torch.tensor(scores).argmax())
    return candidates[best_idx]

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

    encodings = model.tokenizer(
        batch[1],
        return_tensors='pt',
        padding=False,
        truncation=True
    )
    input_ids = encodings['input_ids'].to(device)

    # 모든 후보 생성
    candidates = model.generate_all_candidates(input_ids)

    # MBR Reranking 으로 대표 후보 선정
    decoded_output = model.generate_mbr_from_candidates(candidates)
    
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
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dev_ce.txt") #저장 폴더 이름

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