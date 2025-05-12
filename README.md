# LLM

## 📌 프로젝트 목적
이 프로젝트는 LLM의 기본 작동원리를 알기 위한 실습입니다.

## 📂 사용 데이터셋
- [Harry Potter Books on Kaggle](https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books?select=02+Harry+Potter+and+the+Chamber+of+Secrets.txt)


## 🔧 실행 방법
```bash
# 가상환경 활성화
python -m venv venv
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# GPU 가능 버전 torch 설치는 따로
pip install torch==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
