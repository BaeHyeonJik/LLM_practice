import os
from src.preprocess import clean_text
from src.tokenizer import get_tokens
from src.loader import get_loaders
from src.train import model_train

folder_path = "data/HarryPotter"

# 디렉터리 생성
models_dir = "models"
checkpoints_dir = "checkpoints"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

def main():
  # 1. 전처리
  clean_text(folder_path)
  print("1. 전처리 완료")

  # 2. 토큰화
  tokens_ids = get_tokens(os.path.join(folder_path, "clean"))
  print("2. 토근화 완료")

  # 3. 데이터 로드
  train_loader, val_loader = get_loaders(tokens_ids)
  print("3. 데이터 로드 완료")

  # 4. 훈련
  model_train(train_loader, val_loader, models_dir, checkpoints_dir)


if __name__ == "__main__":
  main()