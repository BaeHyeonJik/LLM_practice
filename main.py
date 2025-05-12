import os
from src.preprocess import clean_text
from src.tokenizer import get_tokens

folder_path = "data/HarryPotter"
def main():
  # 1. 전처리
  if not os.path.exists(os.path.join(folder_path, "clean")):
    clean_text(folder_path)
    print("1. 전처리 완료")
  else:
    print("1. 전처리 이미 완료됨")

  # 2. 토큰화
  tokens = get_tokens(os.path.join(folder_path, "clean"))
  print("2. 토근화 완료")








if __name__ == "__main__":
  main()