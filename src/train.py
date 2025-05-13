import torch
import tokenizer
import tiktoken
from src.model import GPTModel
from tqdm import tqdm


tokenizer = tiktoken.get_encoding("gpt2")

def model_train(train_loader, val_loader):

  # hyperparameter 정의
  vocab_size = tokenizer.n_vocab  # 50257, Tiktoken을 사용할 때 어휘 수
  # VOCAB_SIZE = len(tokenizer)     # AutoTokenizer를 사용할 때 어휘 수
  emb_dim = 768
  context_length = 128  # 문맥 길이 (원래: 1024, 줄여서 사용)
  num_heads = 12  # 멀티헤드 어텐션의 헤드 수
  num_layers = 12  # Transformer 블록의 층 수
  drop_rate = 0.1  # 드롭아웃 비율 (과적합 방지)

  # 모델 정의
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(123)
  model = GPTModel(vocab_size, emb_dim, context_length, num_heads, drop_rate, num_layers).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

  train_losses = []
  tokens_seen, global_step = 0, -1
  best_val_loss = float('inf')

  for epoch in range(100):
    model.train()

    epoch_train_loss = 0
    for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
      optimizer.zero_grad()
      input_batch, target_batch = input_batch.to(device), target_batch.to(device)

      logits = model(input_batch)
      loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

      epoch_train_loss += loss.item()
      loss.backward()
      optimizer.step()

      tokens_seen += input_batch.numel()
      global_step += 1

      if global_step % 1000 == 0:
        print(f"Tokens seen: {tokens_seen}")

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 검증
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
      for input_batch, target_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      torch.save(model.state_dict(), f"best_model_{str(epoch+1).zfill(3)}.pth")
      print("model updated!")

    
    torch.save(model.state_dict(), f"model_{str(epoch+1).zfill(3)}.pth")



    



