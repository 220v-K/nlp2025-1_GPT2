'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation_Prefix.py --use_gpu`

학습 없이 기본 GPT2모델로 추론
forward() 및 generate() 코드 변경

입력 데이터: data/sonnets_held_out_dev.txt
정답 데이터: data/TRUE_sonnets_held_out_dev.txt
추론 결과: predictions/generated_sonnets_dev_Prefix.txt

GPU가 여러장이면 사용할 gpu번호를 아래에 적어주고 없으면 다음 코드를 지우면 된다.
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
'''
import argparse
import random
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from optimizer import AdamW
from evaluation import test_sonnet

# GPU 선택 (CUDA_VISIBLE_DEVICES 환경변수로도 지정 가능)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
TQDM_DISABLE = False

# 체크포인트 디렉토리
CHECKPOINT_DIR = "sonnet_v1"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def seed_everything(seed: int = 11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class SonnetGPT(nn.Module):
    """Prefix-Tuning이 적용된 소넷 생성용 GPT-2 모델"""
    def __init__(self, args):
        super().__init__()
        # 1) GPT-2 본체 불러와서 파라미터 고정
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        for p in self.gpt.parameters():
            p.requires_grad = False

        # 2) Tokenizer 설정
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3) 학습할 연속 프롬프트(prefix) 벡터
        self.prefix_len = args.prefix_len
        self.hidden_size = args.d
        self.prefix_embeddings = nn.Parameter(
            torch.randn(self.prefix_len, self.hidden_size) * 0.02
        )

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [B, T]
        attention_mask: [B, T]
        """
        device     = self.get_device()
        batch_size = input_ids.size(0)

        # 1) 단어+위치 임베딩
        token_embeds = self.gpt.embed(input_ids)                  # [B, T, D]

        # 2) prefix 벡터 복제 및 concat
        prefix       = self.prefix_embeddings.unsqueeze(0)        # [1, P, D]
        prefix       = prefix.expand(batch_size, -1, -1).to(device)  # [B, P, D]
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)  # [B, P+T, D]

        # 3) attention mask 확장
        prefix_mask    = torch.ones(batch_size, self.prefix_len,
                                    device=device,
                                    dtype=attention_mask.dtype)  # [B, P]
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, P+T]

        # 4) Transformer 블록 통과
        hidden_states = self.gpt.encode(inputs_embeds, attention_mask)    # [B, P+T, D]
        hidden_states = self.gpt.final_layer_norm(hidden_states)          # LayerNorm

        # 5) prefix 부분 제거
        hidden_states = hidden_states[:, self.prefix_len :, :]            # [B, T, D]

        # 6) weight tying projection → logits
        logits = self.gpt.hidden_state_to_token(hidden_states)            # [B, T, V]
        return logits

    def get_device(self):
        return next(self.gpt.parameters()).device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
        """
        Prefix를 자동으로 붙여가며 토큰 단위로 생성.
        """
        device = self.get_device()
        token_ids = encoding.to(device)
        attn_mask = torch.ones_like(token_ids)

        init_len = token_ids.size(1)
        for _ in range(max_length):
            logits = self.forward(token_ids, attn_mask)             # [B, L, V]
            logits_last = logits[:, -1, :] / temperature            # [B, V]
            probs = torch.softmax(logits_last, dim=-1)              # [B, V]

            # top-p 샘플링
            s_probs, s_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(s_probs, dim=-1)
            mask = cum <= top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = True
            s_probs = s_probs * mask
            s_probs = s_probs / s_probs.sum(dim=-1, keepdim=True)

            choice = torch.multinomial(s_probs, 1)                  # [B, 1]
            token = s_idx.gather(-1, choice)                       # [B, 1]

            # EOS 토큰 나오면 중단
            if token.item() == self.tokenizer.eos_token_id:
                break

            token_ids = torch.cat([token_ids, token], dim=1)        # [B, L+1]
            attn_mask = torch.cat([attn_mask, torch.ones_like(token)], dim=1)

        text = self.tokenizer.decode(token_ids[0].cpu().tolist())
        if init_len <= 1 and text.startswith(self.tokenizer.eos_token):
            text = text[len(self.tokenizer.eos_token):]
        return token_ids, text

def save_model(model, optimizer, args, filename: str):
    path = os.path.join(CHECKPOINT_DIR, filename)
    info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'rng': torch.get_rng_state()
    }
    torch.save(info, path)
    print(f"[SAVE] {path}")
    return path

@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  args = add_arguments(args)
  model = SonnetGPT(args)  # ✅ checkpoint 불러오지 않고 새로 만듦
  model = model.to(device)
  model.eval()

  ds = SonnetsDataset(args.held_out_sonnet_path)
  with open(args.sonnet_out, 'w') as fw:
    fw.write("--Generated Sonnets--\n\n")
    for sid, prompt in ds:
        enc = model.tokenizer(prompt, return_tensors='pt')
        _, gen = model.generate(enc['input_ids'], temperature=args.temperature, top_p=args.top_p)
        fw.write(f"{sid}\n{gen}\n\n")

def get_args():
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--gold_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--prefix_len", type=int, default=12)

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dev_Prefix.txt") #저장 폴더 이름

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

  generate_submission_sonnets(args)
  generate_sonnets_from_checkpoint(args)