import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
  def __init__(self, token_ids, max_length, stride):
    self.input_ids = []
    self.target_ids = []

    # 슬라이딩 윈도우 방식으로 input과 target 시퀀스를 생성
    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i: i + max_length]
      target_chunk = token_ids[i + 1: i + max_length + 1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  # 전체 샘플 수를 반환 (DataLoader에서 사용)
  def __len__(self):
    return len(self.input_ids)
  
  # 인덱스에 해당하는 input, target 시퀀스 반환
  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
  

def get_loaders(token_ids: list[int]) -> DataLoader:

  # MyDataset 클래스에 token_ids를 전달하여 데이터셋을 만듦
  dataset = MyDataset(token_ids, max_length = 32, stride = 4)

  # DataLoader 객체를 생성
  # dataset: 데이터를 배치 단위로 반환할 MyDataset 객체
  # batch_size: 배치 크기
  # shuffle: 데이터를 섞어서 반환
  # drop_last: 마지막 배치가 배치 크기보다 작으면 버림
  train_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

  return train_loader







