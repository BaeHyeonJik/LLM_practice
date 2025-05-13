import os
import re


def clean_text(folder_path : str):
  
  # clean 폴더 생성
  clean_folder_path = os.path.join(folder_path, "clean")
  print(clean_folder_path)
  if not os.path.exists(clean_folder_path):
    os.makedirs(clean_folder_path)

  origin_folder_path = os.path.join(folder_path, "raw")

  for filename in os.listdir(origin_folder_path):
    clean_filename = "cleaned_" + filename
    if not os.path.exists(os.path.join(clean_folder_path, clean_filename)):
      if filename.endswith('.txt'):
        file_path = os.path.join(origin_folder_path, filename)

        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as file:
          book_text = file.read()

        # 전처리
        clean_text = re.sub(r'\n+', ' ', book_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)

        # 파일 쓰기
        new_filename = os.path.join(clean_folder_path, clean_filename)
        with open(new_filename, 'w', encoding='utf-8') as file:
          file.write(clean_text)

  