import torch
import math

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # key, value, query에 대한 선형변환 layer 초기화.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # 이 드롭아웃은 트랜스포머의 원래 구현에 따라 normalized attention scores에 적용된다.
    # 다소 이례적이지만, 경험적으로 이것이 더 나은 성능을 제공한다고 알려져 있다.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # hidden_state (x) 를 사영하기 위해 k, v, q의 해당 linear_layer가 사용된다.
    proj = linear_layer(x)
    # 다음으로, 프로젝션에 대해 여러 헤드를 생성해야 한다. 
    # 이는 은닉 상태를 self.num_attention_heads로 분할하며, 
    # 각 헤드는 self.attention_head_size 크기를 갖도록 한다.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # 적절히 전치하여 크기 [bs, num_attention_heads, seq_len, attention_head_size]인 프로젝션을 얻는다.
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ##----- 새로 작성한 코드 -----
    # query와 key 전치 행렬 간의 행렬 곱셈을 수행하여 attention score 계산
    # key.size() = [bs, num_attention_heads, seq_len, attention_head_size]
    # query.size() = [bs, num_attention_heads, seq_len, attention_head_size]
    # key.transpose(-1, -2).size() = [bs, num_attention_heads, attention_head_size, seq_len]
    # attention_scores.size() = [bs, num_attention_heads, seq_len, seq_len]
    attention_scores = torch.matmul(query, key.transpose(-1, -2))

    # attention score를 attention_head_size의 제곱근으로 스케일링
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # attention_mask를 적용하여 padding 토큰에 대한 attention을 방지
    # attention_mask.size() = [bs, 1, 1, seq_len]
    # attention_scores.size() = [bs, num_attention_heads, seq_len, seq_len]
    attention_scores = attention_scores + attention_mask

    # attention score에 softmax를 적용하여 확률 분포 생성
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # dropout 적용
    attention_probs = self.dropout(attention_probs)

    # attention 확률과 value 행렬 간의 행렬 곱셈 수행
    # attention_probs.size() = [bs, num_attention_heads, seq_len, seq_len]
    # value.size() = [bs, num_attention_heads, seq_len, attention_head_size]
    # context_layer.size() = [bs, num_attention_heads, seq_len, attention_head_size]
    context_layer = torch.matmul(attention_probs, value)

    # context_layer 전치하여 크기 [bs, seq_len, num_attention_heads, attention_head_size]로 변경
    # context_layer.size() = [bs, seq_len, num_attention_heads, attention_head_size]
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

    # context_layer를 재구성하여 크기 [bs, seq_len, all_head_size]로 변경
    # all_head_size = num_attention_heads * attention_head_size
    # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    # context_layer = context_layer.view(*new_context_layer_shape)
    context_layer = rearrange(context_layer, 'b t h d -> b t (h d)')
    
    return context_layer
    ##-------------------------


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # 먼저, self.transform을 사용하여 multi-head attention에 필요한
    # 각 토큰의 key, value, query를 생성해야 한다(함수 내부에 자세한 내용 있음).
    # *_layer의 크기 = [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # multi-head attention 계산.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
