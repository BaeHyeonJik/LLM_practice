import os
import re


def clean_text(folder_path):

  for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
      file_path = os.path.join(folder_path, filename)

      # 파일 읽기
      with open(file_path, 'r', encoding='utf-8') as file:
        book_text = file.read()

      # 전처리
      clean_text = re.sub(r'\n+', ' ', book_text)
      clean_text = re.sub(r'\s+', ' ', clean_text)

      new_filename = os.path.join(folder_path, "cleaned_" + filename)

      with open(new_filename, 'w', encoding='utf-8') as file:
        file.write(clean_text)

  