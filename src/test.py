import torch
from model import GPTModel
import tiktoken

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    
    for _ in range(max_new_tokens):
        # 입력 토큰을 context로 자름(입력 토큰의 길이가 context_length를 넘기면 안됌)
        idx_cond = idx[:, -context_size:]

        # 마지막 위치의 토큰에 대한 예측만 사용
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 확률이 높은 상위 top_k개의 토큰만 남기고 나머지는 제거
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # temperature > 0: 확률적으로 샘플링 vs temperature == 0: 확률 가장 높은 토큰 선택
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 예측 단어가 eos_id면 종료
        if idx_next == eos_id:
            break

        # 생성한 토큰을 입력 뒤에 붙여서 다시 입력.
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

start_context = input("Start context: ")

# token화 후 배치 차원 추가
tokenizer = tiktoken.get_encoding("gpt2")
idx = tokenizer.encode(start_context)
idx = torch.tensor(idx).unsqueeze(0)

# GPT 모델이 처리할 수 있는 최대 토큰 수
context_size = model.pos_emb.weight.shape[0] 

# 입력 문장 이후의 단어들을 만들어 줌(최대 50 토큰)
token_ids = generate(
    model=model,
    idx=idx.to(device),
    max_new_tokens=50,
    context_size= context_size,
    top_k=50,
    temperature=0.5
)

# 배치 차원 제거 및 토큰 리스트를 문자열로 복원
flat = token_ids.squeeze(0)
out = tokenizer.decode(flat.tolist()).replace("\n", " ")

print("output", ":", out)