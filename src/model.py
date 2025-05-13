import torch
import torch.nn as nn
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# 하이퍼파라미터 정의
VOCAB_SIZE = tokenizer.n_vocab
EMB_DIM = 768
CONTEXT_LENGTH = 128
NUM_HEADS = 12
NUM_LAYERS = 12
DROP_RATE = 0.1

# MultiHeadAttention class 정의
class SelfAttention(nn.Module):
  def __init__(self, embed_dim, atten_dim, drop_rate, context_length):
    super().__init__()

    self.d_out = atten_dim

    # Query, Key, Value 선형 레이어 정의
    # W_query: 내가 어떤 정보를 찾고 싶은지를 표현 (질문 역할)
    # W_key  : 각 토큰이 어떤 정보인지 표현 (정보의 제목)
    # W_value: 실제로 전달할 정보 (실제 내용)
    self.W_query = nn.Linear(embed_dim, atten_dim, bias=False)
    self.W_key = nn.Linear(embed_dim, atten_dim, bias=False)
    self.W_value = nn.Linear(embed_dim, atten_dim, bias=False)

    # 드롭아웃 정의
    self.dropout = nn.Dropout(drop_rate)

    # 마스크 등록 (상삼각 행렬을 사용 => 정답지 차단)
    mask = torch.triu(torch.ones((context_length, context_length)), diagonal=1)
    self.register_buffer('mask', mask)

  def forward(self, x):
    # Query, Key, Value 생성
    b, num_tokens, _ = x.shape
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    # 어텐션 스코어 계산 (QKᵀ)
    attn_scores = queries @ keys.transpose(-2, -1)

    # 마스크 적용(미래 시점의 토근에는 attention 방지)
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    attn_scores.masked_fill_(mask_bool, -torch.inf)

    # 스케일 조정 후 softmax로 어텐션 가중치 계산(atten_dim 기준), 이후 드롭아웃 적용
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    # 컨텍스트 벡터 계산
    context_vec = (attn_weights @ values).transpose(1, 2)
    context_vec = context_vec.reshape(b, num_tokens, self.d_out)

    return context_vec


class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, drop_rate, context_length):
    super().__init__()
    assert embed_dim % num_heads == 0

    # 헤드당 차원 계산
    atten_dim = embed_dim // num_heads

    # 여러 개의 SelfAttention을 사용
    self.attentions = nn.ModuleList([SelfAttention(embed_dim, atten_dim, drop_rate, context_length) for _ in range(num_heads)])

    # 최종 출력 프로젝션
    self.fc = nn.Linear(embed_dim, embed_dim)

  def forward(self, x):
    head_outputs = []

    # 헤드 병합(마지막 차원(atten_dim 기준))
    for attention in self.attentions:
      head_output = attention(x)
      head_outputs.append(head_output)
    concatenated_heads = torch.cat(head_outputs, dim=-1)

    # 최종 프로젝션
    output = self.fc(concatenated_heads)

    return output
  

# LayerNorm class 정의
class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5

    # γ는 처음에는 곱해도 그대로 나와야 하기에 1
    # β는 처음에는 더해도 그대로 나와야 하기에 0
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  # norm_x = (x - μ) / sqrt(σ² + ε)
  # 최종 출력 = γ * norm_x + β
  def forward(self, x):

    # keepdim=True => shape: (batch, seq_len, 1) => broadcasting이 가능하게 유지
    mean = x.mean(dim=-1, keepdim=True)  
    # unbiased=False => 전체데이터의 특성을 그대로 반영 => n-1 이 아니라 n 으로 나눠줌
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    output = self.scale * norm_x + self.shift
    return output
  

# GELU class 정의
class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  # GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
  def forward(self, x):
    output = 0.5 * x * (1 + torch.tanh(
      torch.sqrt(torch.tensor(2.0/torch.pi)) *
      x + 0.044715 * torch.pow(x, 3)))
    return output
  

# FeedForward class 정의
class FeedForward(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(emb_dim, 4 * emb_dim),
      GELU(),
      nn.Linear(4 * emb_dim, emb_dim)
    )

  def forward(self, x):
    output = self.layers(x)
    return output
  

# TransformerBlock class 정의
class TransformerBlock(nn.Module):
  def __init__(self, emb_dim, num_heads, drop_rate, context_length):
    super().__init__()
    self.att = MultiHeadAttention(emb_dim, num_heads, drop_rate, context_length)
    self.ff = FeedForward(emb_dim)
    self.norm1 = LayerNorm(emb_dim)
    self.norm2 = LayerNorm(emb_dim)
    self.drop_shortcut = nn.Dropout(drop_rate)

  # Pre-Norm 구조
  # x → LayerNorm -> Attention -> Dropout -> Residual
  # -> LayerNorm -> FeedForword -> Dropout -> Residual
  def forward(self, x):
    short_cut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + short_cut

    short_cut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    output = x + short_cut

    return output
  

# GPTModel class 정의
class GPTModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
    self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM)
    self.drop_emb = nn.Dropout(DROP_RATE)
    self.trf_blocks = nn.Sequential(*[
        TransformerBlock(EMB_DIM, NUM_HEADS, DROP_RATE, CONTEXT_LENGTH) 
        for _ in range(NUM_LAYERS)
    ])

    self.final_norm = LayerNorm(EMB_DIM)
    self.out_head = nn.Linear(EMB_DIM, VOCAB_SIZE, bias=False)

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape

    # GPT 모델 전체 구현
    # TokenEmbedding + PositionalEmbedding -> Dropout -> TransformerBlock × L
    # -> inal LayerNorm → Linear Projection (Out Head) → Logits 반환
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
    x = tok_embeds + pos_embeds
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits