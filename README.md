# LLM

## 📌 프로젝트 목적
이 프로젝트는 LLM의 기본 작동원리를 알기 위한 실습입니다.

## 📂 사용 데이터셋
- [Harry Potter Books on Kaggle](https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books?select=02+Harry+Potter+and+the+Chamber+of+Secrets.txt)

## 🧠 모델 구조
아래는 구현한 GPT 구조입니다.

![GPT 구조도](https://www.mdpi.com/mathematics/mathematics-11-02320/article_deploy/html/images/mathematics-11-02320-g001.png)


## 🔧 실행 방법
```bash

# 폴더 생성
`root/data/HarryPotter/raw/` 폴더를 생성하고, 원본 데이터를 이곳에 삽입하세요.


# 가상환경 활성화
python -m venv venv
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# GPU 가능 버전 torch 설치는 따로
pip install torch==2.7.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.7.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118


