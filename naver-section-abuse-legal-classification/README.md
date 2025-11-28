# 네이버 뉴스 섹션별 악플 법적 유형 분류 연구

## 📌 프로젝트 개요

네이버 뉴스의 정치/사회/연예 섹션별 댓글에서 악플의 법적 유형 분포를 분석하고, 머신러닝 기반 자동 분류 모델을 비교 연구하는 프로젝트입니다.

### 연구 질문

1. 정치·사회·연예 기사 댓글에서 악플의 **법적 유형 분포**가 어떻게 다른가?
2. 수집한 댓글을 이용해 악플 유형을 **머신러닝 기반으로 자동 분류**할 수 있는가?

### 분류 카테고리 (법적 기준)

- **A. 명예훼손형**: 특정 인물이나 집단의 명예를 훼손하는 내용
- **B. 모욕형**: 공연히 사람을 모욕하는 내용
- **C. 혐오표현형**: 집단 혐오/차별 표현
- **D. 협박·위협형**: 협박이나 위협의 내용
- **E. 성희롱·성폭력형**: 성적 모욕이나 성폭력 관련 표현
- **F. 악플 아님**: 일반 의견/비판

## 📁 프로젝트 구조

```
naver-section-abuse-legal-classification/
├── data/
│   ├── raw/                    # 원본 댓글 데이터
│   ├── processed/              # 전처리 및 라벨링된 데이터
│   └── samples/                # 샘플 데이터
├── notebooks/                  # Jupyter 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_labeling_guideline.ipynb
│   ├── 03_section_distribution_analysis.ipynb
│   └── 04_kobert_training.ipynb
├── src/                        # 소스 코드
│   ├── scrape_comments.py     # 댓글 수집
│   ├── data_loader.py         # 데이터 로더
│   ├── preprocessing.py       # 전처리
│   ├── labeling_rules.py      # 라벨링 규칙
│   ├── feature_extraction.py  # 특징 추출
│   ├── train_model.py         # 모델 학습
│   ├── evaluate.py            # 모델 평가
│   ├── analyze_distribution.py # 분포 분석
│   └── utils.py               # 유틸리티
├── results/                    # 결과 파일
│   ├── charts/                # 그래프
│   ├── logs/                  # 로그
│   └── model/                 # 학습된 모델
├── report/                     # 보고서
│   └── ppt/                   # 발표 자료
├── labeling_guideline.md       # 라벨링 기준 문서
├── requirements.txt            # 패키지 의존성
└── README.md                   # 이 파일
```

## 🚀 설치 및 설정

### 1. 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 디렉토리 생성

```bash
mkdir -p data/raw data/processed data/samples
mkdir -p results/charts results/logs results/model
mkdir -p report/ppt
```

## 📊 사용 방법

### Phase 1: 데이터 수집

네이버 뉴스 기사에서 댓글을 수집합니다.

```bash
python src/scrape_comments.py \
    --url "https://n.news.naver.com/article/..." \
    --section politics \
    --max_clicks 30 \
    --headless
```

**섹션별 수집 목표:**
- 정치/사회/연예 섹션 각각 5개 기사
- 섹션별 500개 댓글 (총 1500개)

### Phase 2: 라벨링 작업

1. `labeling_guideline.md` 파일을 참고하여 라벨링 기준을 확인합니다.
2. 수집한 댓글을 수동으로 라벨링합니다.
3. 라벨링된 데이터를 `data/processed/comments_labeled.csv`에 저장합니다.
4. **자동 라벨링 보조 도구**: 휴리스틱 기반 예비 라벨을 생성하고 사람이 검수합니다.

**CSV 형식:**
```csv
id,section,comment_text,label
politics_001,politics,"[인물]은 정말 멍청한 것 같다",B
society_001,society,"이 기사 내용이 좋은 것 같다",F
...
```

#### 자동 라벨링 실행 (옵션)

```bash
python -m src.auto_label \
    --input data/raw \
    --output data/processed/comments_labeled_auto.csv \
    --min_confidence 0.6
```

- `predicted_label`: 휴리스틱 결과
- `accepted_label`: 신뢰도 임계값 이상일 때 채택된 라벨 (기본값 0.5)
- `rule_matches`: 해당 라벨을 선택한 키워드/정규표현식

출력 CSV를 기반으로 사람이 검토하여 `comments_labeled.csv`를 확정하면 라벨링 시간을 크게 줄일 수 있습니다.

### Phase 3: 섹션별 분포 분석

```bash
python src/analyze_distribution.py
```

**결과물:**
- `results/cross_table.csv`: 섹션 × 라벨 교차테이블
- `results/percentage_table.csv`: 섹션별 라벨 비율 테이블
- `results/charts/section_label_distribution.png`: 분포 그래프
- `results/charts/label_counts_by_section.png`: 히트맵
- `results/distribution_analysis_report.txt`: 분석 리포트

### Phase 4: 모델 학습

#### Baseline 1: TF-IDF + Logistic Regression

```bash
python src/train_model.py --model baseline1
```

#### Baseline 2: KoBERT 임베딩 + SVM

```bash
python src/train_model.py --model baseline2 --device cuda
```

#### Baseline 3: KoBERT Fine-tuning

```bash
python src/train_model.py --model baseline3 \
    --device cuda \
    --batch_size 16 \
    --epochs 3 \
    --lr 2e-5
```

#### 모든 모델 학습

```bash
python src/train_model.py --model all --device cuda
```

### Phase 5: 모델 평가

```bash
python src/evaluate.py --models baseline1 baseline2 baseline3
```

**결과물:**
- `results/model_comparison_test.csv`: 모델 성능 비교 테이블
- `results/charts/cm_*.png`: 각 모델의 Confusion Matrix
- `results/charts/model_comparison_test.png`: 모델 비교 그래프
- `results/evaluation_report.txt`: 평가 리포트

## 📈 실험 결과

### 평가 지표

- **Accuracy**: 전체 정확도
- **Macro F1**: 클래스별 F1 점수의 평균
- **Weighted F1**: 클래스별 샘플 수를 가중치로 한 F1 점수

### 모델 비교

| 모델 | Accuracy | Macro F1 | Weighted F1 |
|------|----------|----------|-------------|
| Baseline 1 (TF-IDF + LR) | - | - | - |
| Baseline 2 (KoBERT + SVM) | - | - | - |
| Baseline 3 (KoBERT Fine-tuning) | - | - | - |

*실제 결과는 데이터 수집 및 학습 후 업데이트됩니다.*

## 📝 라벨링 기준

자세한 라벨링 기준은 `labeling_guideline.md` 파일을 참고하세요.

## 🔬 연구 방법론

### 데이터 분할

- Train/Val/Test = 6:2:2 (Stratified Split)
- 클래스 불균형 대응: class weight 부여

### 전처리

- 이모티콘/반복 문자 정규화
- 욕설 변형 치환 (예: ㅂㅅ → 바보)
- [인물], [집단] 토큰 적용

### 특징 추출

- **Baseline 1**: TF-IDF (uni/bi-gram, 문자 단위)
- **Baseline 2**: KoBERT [CLS] 벡터 (768차원)
- **Baseline 3**: KoBERT Fine-tuning

### 통계 분석

- 섹션별 분포 비교: 교차테이블, 백분율 비교
- 통계적 유의성 검정: 카이제곱 검정

## 📚 참고 자료

- KoBERT: [monologg/kobert](https://huggingface.co/monologg/kobert)
- Transformers: [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## 📄 라이선스

이 프로젝트는 연구 목적으로 작성되었습니다.

## 👥 기여

이슈 및 풀 리퀘스트를 환영합니다.

