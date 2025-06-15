import torch
from torch import nn
import torch.nn.functional as F
from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: forward() 함수를 위한 이 helper 메서드를 구현하시오:
      - 이 함수는 multi-head attention layer와 feed forward layer 이후에 적용된다.
      - GPT-2 layer는 각 sublayer의 변환된 출력에 드롭아웃을 적용한 후, 이를 sublayer 입력에 더한다. 
        이 함수에서는 Layer Normalization을 적용하지 않는다.
    """
    ##----- 새로 작성한 코드 -----
    output = dropout(dense_layer(output))
    output = input + output
    return output
    ##------------------------


  def forward(self, hidden_states, attention_mask):
    """
    TODO: forward pass의 구현. 고려해야 할 주요 사항은 다음과 같다:
      - Multi-head Attention layer(CausalSelfAttention): mask된 입력을 기반으로 self-attention을 계산한다.
      - Layer Normalization: Attention layer와 Feed-forward layer 이전에 적용된다.
      - Dropout, Residual Connection, Layer Normalization를 적용하시오(self.add() 메서드를 사용)
      - Feed-Forward layer: hidden states를 추가로 refine하기 위해 변환을 적용한다.
    """

    """
    Multi-head self-attention → Add & Norm → FFN → Add & Norm (Pre-LN GPT-2)
    """
    ##----- 새로 작성한 코드 -----

    # ─── 1. LayerNorm 후 Self-Attention ──────────────────────────────
    normed_states = self.attention_layer_norm(hidden_states)

    # (a) padding-mask( [bs,1,1,seq] )에
    # (b) causal-mask( [1,1,seq,seq] )를 더해 combined mask 생성
    seq_len = normed_states.size(1)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=normed_states.device))
    causal = causal.unsqueeze(0).unsqueeze(0)                     # [1,1,seq,seq]
    mask_val = torch.finfo(normed_states.dtype).min
    causal = (1.0 - causal) * mask_val                            # 0 / -∞
    attn_mask = attention_mask + causal                           # [bs,1,seq,seq]

    attn_out = self.self_attention(normed_states, attn_mask)      # [bs,seq,h]

    # Add, Dropout, Dense 
    attn_out = self.add(hidden_states, attn_out,
                        self.attention_dense, self.attention_dropout)

    # LayerNorm, Feed-Forward 
    normed_attn = self.out_layer_norm(attn_out)
    ff = self.interm_af(self.interm_dense(normed_attn))
    ff = self.out_dense(ff)

    # Add & Dropout (Dense 이미 적용됨) 
    output = self.add(attn_out, ff, lambda x: x, self.out_dropout)
    return output
    ##------------------------