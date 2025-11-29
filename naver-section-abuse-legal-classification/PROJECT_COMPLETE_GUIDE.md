# 네이버 뉴스 섹션별 악플 법적 유형 분류 연구 - 완전 가이드

## 📋 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [전체 프로세스 요약](#전체-프로세스-요약)
3. [Phase별 상세 진행 내역](#phase별-상세-진행-내역)
4. [코드 파일 구조 및 설명](#코드-파일-구조-및-설명)
5. [결과 파일 위치 및 내용](#결과-파일-위치-및-내용)
6. [실험 결과 상세](#실험-결과-상세)
7. [실행 방법](#실행-방법)
8. [Git 커밋 내역](#git-커밋-내역)

---

## 프로젝트 개요

### 연구 제목
- **한글**: 네이버 뉴스 섹션(정치·사회·연예)별 악플 법적 유형의 분포 차이와 자동분류 모델 비교 연구
- **영문**: Section-Specific Distribution and Automatic Classification of Legal-Type Abusive Comments in Naver News

### 연구 질문
1. 정치·사회·연예 기사 댓글에서 악플의 법적 유형 분포가 어떻게 다른가?
2. 수집한 댓글을 이용해 악플 유형을 머신러닝 기반으로 자동 분류할 수 있는가?

### 데이터 규모
- **총 댓글 수**: 1,500개
- **섹션별**: 정치 500개, 사회 500개, 연예 500개
- **악플 개수**: 64개 (4.3%)
- **악플 유형**: A(명예훼손) 16개, B(모욕) 34개, E(성희롱) 14개

---

## 전체 프로세스 요약

```
1. 데이터 수집 (Phase 1)
   ↓
2. 익명화 및 라벨링 (Phase 2)
   ↓
3. 섹션별 분포 분석 (Phase 3)
   ↓
4. 모델 학습 및 평가 (Phase 4)
   ↓
5. 추가 실험 (섹션별 모델, SMOTE, KoBERT)
   ↓
6. 논문 및 발표 자료 작성 (Phase 5)
```

---

## Phase별 상세 진행 내역

### Phase 1: 데이터 수집

#### 진행 내용
- 네이버 뉴스 기사에서 댓글 자동 수집
- Selenium을 사용한 웹 스크래핑
- 섹션별 500개씩 목표로 수집

#### 사용한 코드 파일
- **`src/scrape_comments.py`**: 단일 기사 댓글 수집 함수
- **`src/scrape_batch.py`**: 여러 기사 배치 수집 스크립트
- **`collect_all_comments.sh`**: 전체 섹션 자동 수집 스크립트

#### 실행 명령어
```bash
# 전체 섹션 자동 수집
./collect_all_comments.sh

# 또는 섹션별 개별 수집
python -m src.scrape_batch --section politics --target_count 500 --headless --urls [URL1] [URL2] [URL3]
```

#### 생성된 데이터 파일
- **`data/raw/comments_politics.csv`**: 정치 섹션 댓글 500개
- **`data/raw/comments_society.csv`**: 사회 섹션 댓글 500개
- **`data/raw/comments_entertainment.csv`**: 연예 섹션 댓글 500개

#### 결과 확인
```bash
# 수집된 데이터 확인
ls -lh data/raw/comments_*.csv
# 각 파일 약 60-90KB 크기
```

---

### Phase 2: 익명화 및 라벨링

#### 진행 내용
- 휴리스틱 규칙 기반 자동 라벨링 시스템 구축
- 법적 기준에 따른 6개 카테고리 분류 (A~F)
- 신뢰도 기반 라벨 필터링

#### 사용한 코드 파일
- **`src/heuristic_labeler.py`**: 라벨별 키워드/정규식 기반 라벨링 규칙
- **`src/auto_label.py`**: 자동 라벨링 실행 스크립트
- **`src/labeling_rules.py`**: 라벨 정의 및 우선순위 관리
- **`src/preprocessing.py`**: 텍스트 전처리 (이모티콘 정규화, 욕설 치환)

#### 실행 명령어
```bash
# 자동 라벨링 실행
python -m src.auto_label --input data/raw --output data/processed/comments_labeled_auto.csv --min_confidence 0.6
```

#### 생성된 파일
- **`data/processed/comments_labeled_auto.csv`**: 자동 라벨링 결과 (신뢰도 포함)
- **`data/processed/comments_labeled.csv`**: 최종 라벨링 데이터 (1500개)
- **`labeling_guideline.md`**: 라벨링 기준 문서

#### 라벨 분포 (자동 라벨링 결과)
- F(악플 아님): 1,436개 (95.7%)
- B(모욕형): 34개 (2.3%)
- A(명예훼손형): 16개 (1.1%)
- E(성희롱형): 14개 (0.9%)
- C(혐오표현형): 0개
- D(협박형): 0개

#### 결과 확인
```bash
# 라벨링된 데이터 확인
head -20 data/processed/comments_labeled.csv
# 컬럼: id, section, comment_text, label, label_confidence, predicted_label, rule_matches
```

---

### Phase 3: 섹션별 분포 분석

#### 진행 내용
- 섹션별 악플 분포 교차테이블 생성
- 카이제곱 검정 수행
- 악플만을 대상으로 한 심화 분석
- 시각화 그래프 생성

#### 사용한 코드 파일
- **`src/analyze_distribution.py`**: 전체 데이터 분포 분석
- **`src/analyze_abusive_only.py`**: 악플만 대상 심화 분석

#### 실행 명령어
```bash
# 전체 분포 분석
python -m src.analyze_distribution

# 악플만 심화 분석
python -m src.analyze_abusive_only
```

#### 생성된 결과 파일

##### 통계 테이블
- **`results/cross_table.csv`**: 섹션 × 라벨 교차테이블
- **`results/percentage_table.csv`**: 섹션별 라벨 비율 (%)
- **`results/distribution_analysis_report.txt`**: 전체 분석 리포트

##### 시각화 그래프
- **`results/charts/section_label_distribution.png`**: 섹션별 라벨 분포 막대 그래프
- **`results/charts/label_counts_by_section.png`**: 섹션별 라벨 개수 히트맵
- **`results/charts/abusive_only_section_distribution.png`**: 악플만의 섹션별 분포
- **`results/charts/abusive_only_section_proportion.png`**: 악플만의 섹션별 비율 (스택 바)
- **`results/charts/abusive_type_by_section.png`**: 악플 유형별 섹션 분포

##### 심화 분석 리포트
- **`results/abusive_only_analysis_report.txt`**: 악플만 대상 상세 분석
- **`results/paper_insights.md`**: 논문용 핵심 인사이트

#### 주요 분석 결과

##### 전체 데이터 기준
- **전체 악플 비율**: 4.3% (64개/1,500개)
- **섹션별 악플 비율**:
  - 정치: 7.2% (36개) ⬆️ 가장 높음
  - 사회: 4.2% (21개)
  - 연예: 1.4% (7개) ⬇️ 가장 낮음
- **카이제곱 검정**: χ² = 22.61, p < 0.001 (유의함)

##### 악플만 대상 분석
- **정치 섹션이 전체 악플의 56.2% 차지** (36개/64개)
- **성희롱형(E)이 연예 섹션에서 전혀 발견되지 않음** (0개)
- **연예 섹션 악플의 85.71%가 모욕형(B)**
- **사회 섹션의 평균 심각도가 가장 높음** (3.43)

#### 결과 확인 방법
```bash
# 교차테이블 확인
cat results/cross_table.csv

# 분석 리포트 확인
cat results/distribution_analysis_report.txt

# 그래프 확인 (이미지 뷰어로 열기)
open results/charts/section_label_distribution.png
```

---

### Phase 4: 모델 학습 및 평가

#### 진행 내용
- 3가지 Baseline 모델 학습
- 성능 평가 및 비교
- Confusion Matrix 생성

#### 사용한 코드 파일
- **`src/train_model.py`**: 모델 학습 스크립트
- **`src/evaluate.py`**: 모델 평가 스크립트
- **`src/feature_extraction.py`**: 특징 추출 (TF-IDF, KoBERT)
- **`src/data_loader.py`**: 데이터 로드 및 분할

#### 실행 명령어

##### 모델 학습
```bash
# 전체 데이터로 Baseline1 학습
python -m src.train_model --model baseline1 --device cpu

# 모든 모델 학습
python -m src.train_model --model all --device cpu

# 특정 섹션만 학습
python -m src.train_model --model baseline1 --section politics --device cpu
```

##### 모델 평가
```bash
# 단일 모델 평가
python -m src.evaluate --models baseline1

# 여러 모델 비교
python -m src.evaluate --models baseline1 baseline2 baseline1_smote
```

#### 학습된 모델 파일
- **`results/model/baseline1.pkl`**: TF-IDF + Logistic Regression (346KB)
- **`results/model/baseline2.pkl`**: KoBERT 임베딩 + SVM (24KB)
- **`results/model/baseline3.pkl`**: KoBERT Fine-tuning (352MB)
- **`results/model/kobert_finetune_best.pt`**: KoBERT 최고 성능 모델 (352MB)

#### 평가 결과 파일

##### 성능 리포트
- **`results/evaluation_report.txt`**: 모델별 상세 평가 결과
- **`results/model_comparison_train.csv`**: Train 세트 성능 비교
- **`results/model_comparison_val.csv`**: Validation 세트 성능 비교
- **`results/model_comparison_test.csv`**: Test 세트 성능 비교

##### Confusion Matrix
- **`results/charts/cm_baseline1_train.png`**: Baseline1 Train CM
- **`results/charts/cm_baseline1_val.png`**: Baseline1 Val CM
- **`results/charts/cm_baseline1_test.png`**: Baseline1 Test CM
- **`results/charts/cm_baseline2_*.png`**: Baseline2 CM (train/val/test)
- **`results/charts/cm_baseline1_politics_*.png`**: 정치 섹션 전용 모델 CM
- **`results/charts/cm_baseline1_society_*.png`**: 사회 섹션 전용 모델 CM
- **`results/charts/cm_baseline1_entertainment_*.png`**: 연예 섹션 전용 모델 CM
- **`results/charts/cm_baseline1_smote_*.png`**: SMOTE 적용 모델 CM

##### 성능 비교 그래프
- **`results/charts/model_comparison_train.png`**: Train 세트 비교 그래프
- **`results/charts/model_comparison_val.png`**: Val 세트 비교 그래프
- **`results/charts/model_comparison_test.png`**: Test 세트 비교 그래프

#### 모델 성능 결과

##### Baseline 1 (TF-IDF + Logistic Regression)
| 데이터셋 | Accuracy | Macro F1 | Weighted F1 |
|---------|----------|----------|-------------|
| Train | 99.78% | 97.70% | 99.79% |
| Val | 96.33% | 43.28% | 94.95% |
| Test | 96.33% | 43.28% | 94.95% |

##### Baseline 2 (KoBERT 임베딩 + SVM)
| 데이터셋 | Accuracy | Macro F1 | Weighted F1 |
|---------|----------|----------|-------------|
| Train | 100.00% | 100.00% | 100.00% |
| Val | 94.67% | 24.32% | 93.05% |
| Test | 95.00% | 24.36% | 93.21% |

#### 결과 확인 방법
```bash
# 평가 리포트 확인
cat results/evaluation_report.txt

# 모델 비교 테이블 확인
cat results/model_comparison_test.csv

# Confusion Matrix 확인
open results/charts/cm_baseline1_test.png
```

---

### Phase 5: 추가 실험

#### 실험 1: 섹션별 전용 모델 학습

##### 목적
섹션별 특화 모델이 전체 데이터 모델보다 성능이 우수한지 검증

##### 실행 명령어
```bash
# 정치 섹션 전용 모델
python -m src.train_model --model baseline1 --section politics --device cpu

# 사회 섹션 전용 모델
python -m src.train_model --model baseline1 --section society --device cpu

# 연예 섹션 전용 모델
python -m src.train_model --model baseline1 --section entertainment --device cpu

# 평가
python -m src.evaluate --models baseline1_politics baseline1_society baseline1_entertainment
```

##### 생성된 모델 파일
- **`results/model/baseline1_politics.pkl`**: 정치 섹션 전용 모델
- **`results/model/baseline1_society.pkl`**: 사회 섹션 전용 모델
- **`results/model/baseline1_entertainment.pkl`**: 연예 섹션 전용 모델

##### 결과
| 모델 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|------|--------------|---------------|------------------|
| baseline1 (전체) | 96.33% | 43.28% | 94.95% |
| baseline1_politics | 94.00% | 34.22% | 91.69% |
| baseline1_society | 96.00% | 24.49% | 94.04% |
| **baseline1_entertainment** | **99.00%** | **49.75%** | **98.50%** |

**핵심 발견**: 연예 섹션 전용 모델이 최고 성능 (Macro F1 +6.47%p 향상)

##### 결과 확인
- 리포트: `results/additional_experiments_summary.md` (실험 1 섹션)
- 비교 테이블: `results/model_comparison_test.csv`
- 그래프: `results/charts/model_comparison_test.png`

---

#### 실험 2: 클래스 불균형 대응 기법 (SMOTE)

##### 목적
SMOTE 오버샘플링을 통해 소수 클래스(악플) 탐지 성능 개선

##### 실행 명령어
```bash
# SMOTE 적용 모델 학습
python -m src.train_model --model baseline1 --resample_strategy smote --device cpu

# 비교 평가
python -m src.evaluate --models baseline1 baseline1_smote
```

##### 생성된 모델 파일
- **`results/model/baseline1_smote.pkl`**: SMOTE 적용 모델

##### 결과
| 모델 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|------|--------------|---------------|------------------|
| baseline1 (기본) | 96.33% | **43.28%** | 94.95% |
| baseline1_smote | 96.00% | 36.99% | 94.21% |

**핵심 발견**: SMOTE 적용 시 Macro F1 성능 저하 (-6.29%p)
- 소규모 데이터셋(악플 64개)에서 SMOTE가 오히려 노이즈 생성

##### 결과 확인
- 리포트: `results/additional_experiments_summary.md` (실험 2 섹션)
- 비교 테이블: `results/model_comparison_test.csv`
- Confusion Matrix: `results/charts/cm_baseline1_smote_test.png`

---

#### 실험 3: KoBERT 모델 재현성 확보

##### 목적
KoBERT 기반 모델의 저장/로드 문제 해결 및 성능 평가

##### 개선 사항
- Baseline2: KoBERT 임베딩을 별도로 저장하여 pickle 의존성 제거
- Baseline3: 모델 state dict 저장 방식 개선

##### 실행 명령어
```bash
# Baseline2 재학습 (개선된 저장 방식)
python -m src.train_model --model baseline2 --device cpu

# 평가
python -m src.evaluate --models baseline2
```

##### 생성된 모델 파일
- **`results/model/baseline2.pkl`**: 재학습된 KoBERT + SVM 모델 (24KB)

##### 결과
| 모델 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|------|--------------|---------------|------------------|
| baseline1 (TF-IDF) | 96.33% | **43.28%** | 94.95% |
| baseline2 (KoBERT) | 95.00% | 24.36% | 93.21% |

**핵심 발견**: TF-IDF + LR이 KoBERT + SVM보다 우수한 성능 (+18.92%p)
- 소규모 데이터셋에서는 전통적 방법론이 더 효과적

##### 결과 확인
- 리포트: `results/additional_experiments_summary.md` (실험 3 섹션)
- 평가 리포트: `results/evaluation_report.txt`
- Confusion Matrix: `results/charts/cm_baseline2_test.png`

---

### Phase 6: 논문 및 발표 자료 작성

#### 생성된 문서

##### 논문 초안
- **`report/paper_draft.md`**: 완전한 논문 초안
  - 서론, 관련 연구, 방법론, 실험 결과, 논의, 결론 포함
  - 모든 실험 결과 반영

##### 발표 자료
- **`report/ppt/presentation_outline.md`**: 발표 슬라이드 개요 (12장)
  - 연구 배경, 방법론, 결과, 결론 포함

##### 요약 리포트
- **`results/final_summary.md`**: 전체 프로젝트 요약
- **`results/paper_insights.md`**: 논문용 핵심 인사이트
- **`results/additional_experiments_summary.md`**: 추가 실험 상세 분석
- **`results/final_experiments_comparison.md`**: 최종 모델 비교 테이블

#### 문서 확인 방법
```bash
# 논문 초안 확인
cat report/paper_draft.md

# 발표 개요 확인
cat report/ppt/presentation_outline.md

# 최종 요약 확인
cat results/final_summary.md
```

---

## 코드 파일 구조 및 설명

### 전체 프로젝트 구조
```
naver-section-abuse-legal-classification/
├── data/                          # 데이터 파일
│   ├── raw/                       # 원본 댓글 데이터
│   │   ├── comments_politics.csv      (500개)
│   │   ├── comments_society.csv       (500개)
│   │   └── comments_entertainment.csv (500개)
│   └── processed/                  # 전처리 및 라벨링된 데이터
│       ├── comments_labeled_auto.csv  (자동 라벨링 결과)
│       └── comments_labeled.csv       (최종 라벨링 데이터, 1500개)
│
├── src/                           # 소스 코드
│   ├── scrape_comments.py         # 단일 기사 댓글 수집
│   ├── scrape_batch.py            # 여러 기사 배치 수집
│   ├── auto_label.py              # 자동 라벨링 실행
│   ├── heuristic_labeler.py      # 휴리스틱 라벨링 규칙
│   ├── labeling_rules.py           # 라벨 정의 및 우선순위
│   ├── preprocessing.py           # 텍스트 전처리
│   ├── data_loader.py             # 데이터 로드 및 분할
│   ├── analyze_distribution.py    # 전체 분포 분석
│   ├── analyze_abusive_only.py    # 악플만 심화 분석
│   ├── feature_extraction.py      # 특징 추출 (TF-IDF, KoBERT)
│   ├── train_model.py             # 모델 학습
│   ├── evaluate.py                # 모델 평가
│   └── utils.py                   # 유틸리티 함수
│
├── results/                       # 결과 파일
│   ├── charts/                    # 시각화 그래프 (20개 이상)
│   ├── model/                     # 학습된 모델 (8개)
│   ├── *.csv                      # 통계 테이블
│   ├── *.txt                      # 분석 리포트
│   └── *.md                       # 요약 문서
│
├── report/                        # 논문 및 발표 자료
│   ├── paper_draft.md             # 논문 초안
│   └── ppt/
│       └── presentation_outline.md # 발표 개요
│
├── notebooks/                     # Jupyter 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_labeling_guideline.ipynb
│   ├── 03_section_distribution_analysis.ipynb
│   └── 04_kobert_training.ipynb
│
├── collect_all_comments.sh        # 전체 섹션 수집 스크립트
├── run_analysis.sh                # 분포 분석 실행
├── run_training.sh                 # 모델 학습 실행
├── run_evaluation.sh               # 모델 평가 실행
├── labeling_guideline.md          # 라벨링 기준 문서
├── README.md                      # 프로젝트 개요
├── README_COLLECTION.md           # 수집 가이드
└── requirements.txt               # 패키지 의존성
```

### 데이터 수집 모듈
```
src/
├── scrape_comments.py      # 단일 기사 댓글 수집 함수
├── scrape_batch.py         # 여러 기사 배치 수집 스크립트
└── collect_all_comments.sh # 전체 섹션 자동 수집 스크립트
```

**주요 함수**:
- `scrape_naver_comments()`: 네이버 뉴스 댓글 수집
- `collect_comments_by_section()`: 섹션별 목표 개수 수집
- `save_comments_to_csv()`: 수집된 댓글을 CSV로 저장

### 데이터 처리 모듈
```
src/
├── data_loader.py          # 데이터 로드 및 분할
├── preprocessing.py        # 텍스트 전처리
├── labeling_rules.py        # 라벨 정의 및 규칙
├── heuristic_labeler.py    # 휴리스틱 라벨링 규칙
└── auto_label.py           # 자동 라벨링 실행
```

**주요 함수**:
- `load_labeled_comments()`: 라벨링된 데이터 로드
- `split_data()`: Train/Val/Test 분할 (Stratified)
- `preprocess_text()`: 텍스트 전처리 파이프라인
- `HeuristicLabeler.label()`: 휴리스틱 라벨링

### 분석 모듈
```
src/
├── analyze_distribution.py    # 전체 분포 분석
└── analyze_abusive_only.py   # 악플만 심화 분석
```

**주요 함수**:
- `create_cross_table()`: 교차테이블 생성
- `chi_square_test()`: 카이제곱 검정
- `analyze_abusive_severity()`: 악플 심각도 분석
- `generate_detailed_report()`: 상세 분석 리포트 생성

### 모델 학습 모듈
```
src/
├── feature_extraction.py   # 특징 추출 (TF-IDF, KoBERT)
├── train_model.py         # 모델 학습 스크립트
└── evaluate.py            # 모델 평가 스크립트
```

**주요 함수**:
- `TfidfFeatureExtractor`: TF-IDF 특징 추출
- `KoBERTFeatureExtractor`: KoBERT 임베딩 추출
- `train_baseline1_tfidf_lr()`: Baseline1 학습
- `train_baseline2_kobert_svm()`: Baseline2 학습
- `train_baseline3_kobert_finetune()`: Baseline3 학습
- `evaluate_model()`: 모델 평가
- `compare_models()`: 모델 비교

### 유틸리티 모듈
```
src/
└── utils.py               # 유틸리티 함수
```

---

## 결과 파일 위치 및 내용

### 데이터 파일
```
data/
├── raw/
│   ├── comments_politics.csv      # 정치 섹션 원본 댓글 (500개)
│   ├── comments_society.csv       # 사회 섹션 원본 댓글 (500개)
│   └── comments_entertainment.csv # 연예 섹션 원본 댓글 (500개)
└── processed/
    ├── comments_labeled_auto.csv   # 자동 라벨링 결과 (신뢰도 포함)
    └── comments_labeled.csv       # 최종 라벨링 데이터 (1500개)
```

### 분석 결과 파일
```
results/
├── cross_table.csv                    # 섹션 × 라벨 교차테이블
├── percentage_table.csv               # 섹션별 라벨 비율 (%)
├── distribution_analysis_report.txt    # 전체 분포 분석 리포트
├── abusive_only_analysis_report.txt   # 악플만 심화 분석 리포트
├── evaluation_report.txt              # 모델 평가 리포트
├── model_comparison_train.csv         # Train 세트 모델 비교
├── model_comparison_val.csv           # Val 세트 모델 비교
├── model_comparison_test.csv          # Test 세트 모델 비교
├── final_summary.md                   # 전체 프로젝트 요약
├── paper_insights.md                  # 논문용 핵심 인사이트
├── additional_experiments_summary.md  # 추가 실험 상세 분석
└── final_experiments_comparison.md    # 최종 모델 비교 테이블
```

### 시각화 그래프
```
results/charts/
├── section_label_distribution.png           # 섹션별 라벨 분포 막대 그래프
├── label_counts_by_section.png              # 섹션별 라벨 개수 히트맵
├── abusive_only_section_distribution.png    # 악플만의 섹션별 분포
├── abusive_only_section_proportion.png      # 악플만의 섹션별 비율 (스택 바)
├── abusive_type_by_section.png              # 악플 유형별 섹션 분포
├── cm_baseline1_train.png                   # Baseline1 Train Confusion Matrix
├── cm_baseline1_val.png                     # Baseline1 Val Confusion Matrix
├── cm_baseline1_test.png                    # Baseline1 Test Confusion Matrix
├── cm_baseline2_*.png                       # Baseline2 CM (train/val/test)
├── cm_baseline1_politics_*.png              # 정치 섹션 전용 모델 CM
├── cm_baseline1_society_*.png               # 사회 섹션 전용 모델 CM
├── cm_baseline1_entertainment_*.png         # 연예 섹션 전용 모델 CM
├── cm_baseline1_smote_*.png                 # SMOTE 적용 모델 CM
├── model_comparison_train.png               # Train 세트 모델 비교 그래프
├── model_comparison_val.png                 # Val 세트 모델 비교 그래프
└── model_comparison_test.png                # Test 세트 모델 비교 그래프
```

### 학습된 모델 파일
```
results/model/
├── baseline1.pkl                    # TF-IDF + LR (전체 데이터) - 346KB
├── baseline2.pkl                     # KoBERT + SVM - 24KB
├── baseline3.pkl                     # KoBERT Fine-tuning - 352MB
├── kobert_finetune_best.pt          # KoBERT 최고 성능 모델 - 352MB
├── baseline1_politics.pkl            # 정치 섹션 전용 모델
├── baseline1_society.pkl             # 사회 섹션 전용 모델
├── baseline1_entertainment.pkl       # 연예 섹션 전용 모델
└── baseline1_smote.pkl               # SMOTE 적용 모델
```

### 문서 파일
```
report/
├── paper_draft.md                    # 논문 초안 (완전한 버전)
└── ppt/
    └── presentation_outline.md        # 발표 슬라이드 개요 (12장)

labeling_guideline.md                  # 라벨링 기준 문서
README.md                              # 프로젝트 개요 및 사용법
README_COLLECTION.md                   # 댓글 수집 가이드
```

---

## 실험 결과 상세

### 최종 모델 성능 순위 (Test Set, Macro F1 기준)

| 순위 | 모델 | Accuracy | Macro F1 | Weighted F1 | 특징 |
|------|------|----------|----------|-------------|------|
| 🥇 1위 | **baseline1_entertainment** | **99.00%** | **49.75%** | **98.50%** | 연예 섹션 전용 |
| 🥈 2위 | baseline1 | 96.33% | 43.28% | 94.95% | 전체 데이터 |
| 🥉 3위 | baseline1_smote | 96.00% | 36.99% | 94.21% | SMOTE 적용 |
| 4위 | baseline1_politics | 94.00% | 34.22% | 91.69% | 정치 섹션 전용 |
| 5위 | baseline1_society | 96.00% | 24.49% | 94.04% | 사회 섹션 전용 |
| 6위 | baseline2 | 95.00% | 24.36% | 93.21% | KoBERT + SVM |

### 핵심 발견사항

#### 1. 섹션별 특성에 따른 모델 전략 차별화 필요
- **연예 섹션**: 섹션별 전용 모델이 매우 효과적 (Macro F1 +6.47%p)
- **정치/사회 섹션**: 전체 데이터 모델이 더 우수

#### 2. 소규모 데이터셋에서의 불균형 대응 기법 한계
- SMOTE는 비효과적 (Macro F1 -6.29%p)
- 클래스 가중치 조정이 더 효과적

#### 3. 전통적 방법론의 우수성
- TF-IDF + LR이 KoBERT + SVM보다 우수 (Macro F1 +18.92%p)
- 소규모 데이터셋에서는 단순한 방법론이 더 효과적

#### 4. 최고 성능 달성
- 연예 섹션 전용 모델: Macro F1 49.75%, Accuracy 99.00%

---

## 실행 방법

### 전체 파이프라인 실행

#### 1. 데이터 수집
```bash
cd naver-section-abuse-legal-classification
./collect_all_comments.sh
```

#### 2. 자동 라벨링
```bash
python -m src.auto_label --input data/raw --output data/processed/comments_labeled_auto.csv --min_confidence 0.6
```

#### 3. 분포 분석
```bash
# 전체 분포 분석
python -m src.analyze_distribution

# 악플만 심화 분석
python -m src.analyze_abusive_only
```

#### 4. 모델 학습
```bash
# 전체 데이터로 Baseline1 학습
python -m src.train_model --model baseline1 --device cpu

# 모든 모델 학습
python -m src.train_model --model all --device cpu
```

#### 5. 모델 평가
```bash
# 단일 모델 평가
python -m src.evaluate --models baseline1

# 여러 모델 비교
python -m src.evaluate --models baseline1 baseline2 baseline1_smote
```

### 추가 실험 실행

#### 섹션별 전용 모델
```bash
# 각 섹션별로 학습
python -m src.train_model --model baseline1 --section politics --device cpu
python -m src.train_model --model baseline1 --section society --device cpu
python -m src.train_model --model baseline1 --section entertainment --device cpu

# 평가
python -m src.evaluate --models baseline1_politics baseline1_society baseline1_entertainment
```

#### SMOTE 실험
```bash
# SMOTE 적용 모델 학습
python -m src.train_model --model baseline1 --resample_strategy smote --device cpu

# 비교 평가
python -m src.evaluate --models baseline1 baseline1_smote
```

---

## Git 커밋 내역

### 초기 커밋
- **커밋 해시**: `300afad`
- **메시지**: "feat: build abusive comment analysis pipeline"
- **내용**: 전체 파이프라인 구축 (데이터 수집, 라벨링, 분석, 모델 학습)

### 추가 실험 커밋
- **커밋 해시**: `f6c7528`
- **메시지**: "feat: add 3 additional experiments (section-specific models, SMOTE, KoBERT reproducibility)"
- **내용**: 3가지 추가 실험 결과 및 분석 리포트

### 저장소
- **GitHub**: https://github.com/5seoyoung/naver-section-abuse-legal-classification
- **브랜치**: main

---

## 결과 확인 체크리스트

### 데이터 확인
- [ ] `data/raw/comments_*.csv` - 수집된 원본 댓글 확인
- [ ] `data/processed/comments_labeled.csv` - 라벨링된 데이터 확인

### 분석 결과 확인
- [ ] `results/cross_table.csv` - 교차테이블 확인
- [ ] `results/distribution_analysis_report.txt` - 분포 분석 리포트 확인
- [ ] `results/abusive_only_analysis_report.txt` - 악플 심화 분석 확인
- [ ] `results/charts/*.png` - 모든 시각화 그래프 확인

### 모델 성능 확인
- [ ] `results/evaluation_report.txt` - 모델 평가 리포트 확인
- [ ] `results/model_comparison_test.csv` - 최종 모델 비교 테이블 확인
- [ ] `results/charts/cm_*.png` - 모든 Confusion Matrix 확인
- [ ] `results/charts/model_comparison_*.png` - 모델 비교 그래프 확인

### 문서 확인
- [ ] `report/paper_draft.md` - 논문 초안 확인
- [ ] `report/ppt/presentation_outline.md` - 발표 개요 확인
- [ ] `results/final_summary.md` - 전체 요약 확인
- [ ] `results/final_experiments_comparison.md` - 최종 실험 비교 확인

---

## 프로젝트 완성도

### ✅ 완료된 항목
1. 데이터 수집 (1,500개 댓글)
2. 자동 라벨링 시스템 구축
3. 섹션별 분포 분석 (전체 + 악플만)
4. 3가지 Baseline 모델 학습 및 평가
5. 섹션별 전용 모델 실험
6. SMOTE 불균형 대응 실험
7. KoBERT 모델 재현성 확보
8. 논문 초안 작성
9. 발표 자료 개요 작성
10. 모든 결과 리포트 생성

### 📊 생성된 파일 통계
- **코드 파일**: 15개
- **데이터 파일**: 5개
- **모델 파일**: 8개
- **분석 리포트**: 8개
- **시각화 그래프**: 20개 이상
- **문서 파일**: 5개

### 🎯 핵심 성과
1. **최고 성능 모델**: 연예 섹션 전용 모델 (Macro F1 49.75%)
2. **주요 발견**: 섹션별 특성에 따른 모델 전략 차별화 필요
3. **논문 준비**: 완전한 논문 초안 및 발표 자료 준비 완료

---

## 문의 및 추가 정보

프로젝트의 모든 파일과 결과는 위 경로에서 확인할 수 있습니다.
각 단계별 상세 내용은 해당 섹션의 리포트 파일을 참고하세요.

**핵심 결과 파일**:
- 전체 요약: `results/final_summary.md`
- 실험 비교: `results/final_experiments_comparison.md`
- 논문 초안: `report/paper_draft.md`

