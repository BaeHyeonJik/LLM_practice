import os
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

def get_tokens(folder_path: str) -> list[int]:

  # 모든 txt 파일 합치기
  all_book_text = ''

  for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
      file_path = os.path.join(folder_path, filename)

      # 파일 읽기
      with open(file_path, 'r', encoding='utf-8') as file:
        all_book_text += file.read()

  # token화
  tokens_ids = tokenizer.encode(all_book_text)

  return tokens_ids
  

