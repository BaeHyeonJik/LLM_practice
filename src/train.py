import os
import torch
import tiktoken
from tqdm import tqdm
from src.model import GPTModel

def model_train(train_loader, val_loader, models_dir, checkpoints_dir):

    # 모델 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel().to(device)

    # 가중치 업데이트 optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    best_val_loss = float('inf')
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(100):
        model.train()
        epoch_train_loss = 0

        # 훈련 단계
        for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            
            # 이전 gradient를 초기화(정확한 학습을 위해)
            optimizer.zero_grad()
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            logits = model(input_batch)

            # [batch_size * seq_len, vocab_size] 랑 [batch_size * seq_len] 비교 => loss(scalar)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)

        # 검증 단계
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for input_batch, target_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                logits = model(input_batch)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        epoch_val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 모델 성능이 개선되었을 경우, 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(models_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model improved! Saved to {model_path}")

        # 매 에폭마다 체크포인트 저장
        checkpoints_path = os.path.join(checkpoints_dir, f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoints_path)
        print(f"Checkpoint saved at {checkpoints_path}")
