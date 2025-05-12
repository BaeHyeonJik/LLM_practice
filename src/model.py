import torch
import torch.nn as nn

# VOCAB_SIZE = tokenizer.n_vocab  # 50257, Tiktoken을 사용할 때 어휘 수
# VOCAB_SIZE = len(tokenizer)     # AutoTokenizer를 사용할 때 어휘 수
CONTEXT_LENGTH = 128  # 문맥 길이 (원래: 1024, 줄여서 사용)
NUM_HEADS = 12  # 멀티헤드 어텐션의 헤드 수
NUM_LAYERS = 12  # Transformer 블록의 층 수
DROP_RATE = 0.1  # 드롭아웃 비율 (과적합 방지)
QKV_BIAS = False  # Query/Key/Value 선형 변환 시 bias 사용 여부


class SelfAttention(nn.Module):
  def __init__(self, embed_dim, atten_dim):
    super().__init__()

    self.d_out = atten_dim

    # Query, Key, Value 선형 레이어 정의
    # W_query: 내가 어떤 정보를 찾고 싶은지를 표현 (질문 역할)
    # W_key  : 각 토큰이 어떤 정보인지 표현 (정보의 제목)
    # W_value: 실제로 전달할 정보 (실제 내용)
    self.W_query = nn.Linear(embed_dim, atten_dim, bias=QKV_BIAS)
    self.W_key = nn.Linear(embed_dim, atten_dim, bias=QKV_BIAS)
    self.W_value = nn.Linear(embed_dim, atten_dim, bias=QKV_BIAS)

    # 드롭아웃 정의
    self.dropout = nn.Dropout(DROP_RATE)

    # 마스크 등록 (상삼각 행렬을 사용 => 정답지 차단)
    self.register_buffer('mask', torch.triu(torch.ones((CONTEXT_LENGTH, CONTEXT_LENGTH)), diagonal = 1))

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

    # 스케일 조정 후 softmax로 어텐션 가중치 계산, 이후 드롭아웃 적용
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    # 컨텍스트 벡터 계산
    context_vec = (attn_weights @ values).transpose(1, 2)
    context_vec = context_vec.reshape(b, num_tokens, self.d_out)

    return context_vec


class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super().__init__()
    assert embed_dim % num_heads == 0

    # 헤드당 차원 계산
    atten_dim = embed_dim // num_heads

    # 여러 개의 SelfAttention을 사용
    self.attentions = nn.ModuleList([SelfAttention(embed_dim, atten_dim) for _ in range(num_heads)])

    # 최종 출력 프로젝션
    self.fc = nn.Linear(embed_dim, embed_dim)

  def forward(self, x):
    head_outputs = []

    # 헤드 병합
    for attention in self.attentions:
      head_output = attention(x)
      head_outputs.append(head_output)
    concatenated_heads = torch.cat(head_outputs, dim=-1)

    # 최종 프로젝션
    output = self.fc(concatenated_heads)

    return output


