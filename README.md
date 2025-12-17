# 문화콘텐츠와 자연어처리 프로젝트

## 📌 한국어 반어법 탐지
**부제:** KoCoSa 데이터셋을 기반으로 한 한국어 반어법 특징 분석 및 딥러닝 기반 탐지 시스템 구축

---

## 프로젝트 개요
본 프로젝트는 **KoCoSa(Korean Contextual Sarcasm) 데이터셋**을 활용하여 한국어 반어법(sarcasm)의 **언어적·맥락적 특징을 분석**하고, 이를 바탕으로 **딥러닝 기반 반어법 탐지 모델**을 구축하는 것을 목표로 합니다.

**사용 데이터셋:** KoCoSa (Korean Contextual Sarcasm Dataset)  
🔗 https://huggingface.co/datasets/YuminKim/KoCoSa

특히 반어법이 **문맥(Context)** 과 **응답(Response)** 간의 의미적 불일치에서 자주 발생한다는 점에 주목하여,
- **Response 단독 기반 모델**과
- **Context + Response 결합 기반 모델**
을 비교·분석하였습니다.

모델 측면에서는 **GPT 계열 모델**과 **BERT 계열 모델**을 활용하여, 언어학적 분석 결과가 실제 탐지 성능 향상으로 이어지는지를 검증합니다.

---

## 모델 학습 방식
- **GPT 모델**: 로컬 환경에서 학습 및 평가 수행
- **BERT 모델**: Jupyter Notebook(`.ipynb`) 환경에서 학습 및 평가 수행
  - Google Colab의 **T4 GPU** 활용

---

## 디렉토리 구조 및 파일 설명

```
.
├── Assists/   
│   └── 반어법 표현에 대한 언어학적 분석 결과 표 모음
│
├── Bash/
│   └── GPT 모델 학습 및 평가를 위한 실행 명령어(.sh) 파일 모음
│
├── Configs/
│   └── (GPT) 딥러닝 모델 학습을 위한 argument parsing 관련 설정 코드
│
├── Scripts/
│   ├── analysis.py
│   │   └── 언어학적 분석을 수행하기 위한 메인 실행 코드
│   ├── BERT(Context+Response).ipynb
│   │   └── 전처리 + Context & Response 기반 BERT 학습 및 평가 코드
│   └── BERT(Only Response).ipynb
│       └── 전처리 + Response 단독 기반 BERT 학습 및 평가 코드
│
├── src/
│   ├── data/
│   │   └── KoCoSa_json/
│   │       └── KoCoSa 데이터셋(JSON 형식)
│   └── 딥러닝 학습 및 언어학적 분석을 위한 공통 모듈
│       └── 모델 정의, 데이터 로딩, 다각적 언어 분석 클래스 포함
│
└── requirements.txt
```

---

## 환경 설정

```bash
conda create -n langcont
conda activate langcont
pip install -r requirements.txt
```

---

## 실행 방법

### 1️⃣ BERT 모델 학습 및 평가
- `.ipynb` 파일을 통해 실행
- Google Colab 환경에서 **T4 GPU** 사용

```text
./scripts/BERT(Context+Response).ipynb
./scripts/BERT(Only Response).ipynb
```

---

### 2️⃣ GPT 모델 학습
- 로컬 환경에서 실행  
- Bash 스크립트를 통해 학습 및 평가 진행  

```bash
# Few-shot prompt 기반 실험
bash ./bash/train_gpt_prompt.sh

# Classifier 기반 GPT 모델 학습
bash ./bash/train_gpt.sh
```

---

### 3️⃣ 언어학적 분석 실행

```bash
python ./scripts/analysis.py
```

본 분석은 반어법 문장의 **감정 신호, 의미적 불일치, 맥락 의존성** 등을 정량적으로 분석하며,
모델 설계 및 성능 해석의 근거로 활용됩니다.

### 4️⃣ 영화 리뷰 데이터 크롤링 및 추론 실행