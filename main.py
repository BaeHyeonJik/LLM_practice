import os
from src.preprocess import clean_text

def main():
  # 1. 전처리
  folder_path = "data/HarryPotter"
  clean_text(folder_path)
  print("1. 전처리 완료")









if __name__ == "__main__":
  main()