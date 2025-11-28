# 댓글 수집 가이드

## 📌 수집 목표

- **정치 섹션**: 500개
- **사회 섹션**: 500개
- **연예 섹션**: 500개
- **총합**: 1,500개

## 🚀 빠른 시작

### 방법 1: 전체 자동 수집 (권장)

```bash
./collect_all_comments.sh
```

이 스크립트는 모든 섹션의 댓글을 자동으로 수집합니다.

### 방법 2: 섹션별 개별 수집

#### 정치 섹션
```bash
python -m src.scrape_batch \
    --section politics \
    --target_count 500 \
    --max_clicks 30 \
    --headless \
    --urls \
    https://n.news.naver.com/article/052/0002277256 \
    https://n.news.naver.com/article/011/0004559400?ntype=RANKING \
    https://n.news.naver.com/article/052/0002277544?ntype=RANKING
```

#### 사회 섹션
```bash
python -m src.scrape_batch \
    --section society \
    --target_count 500 \
    --max_clicks 30 \
    --headless \
    --urls \
    https://n.news.naver.com/article/011/0004559349?ntype=RANKING \
    https://n.news.naver.com/article/079/0004089120?ntype=RANKING \
    https://n.news.naver.com/article/081/0003594894?ntype=RANKING
```

#### 연예 섹션
```bash
python -m src.scrape_batch \
    --section entertainment \
    --target_count 500 \
    --max_clicks 30 \
    --headless \
    --urls \
    https://n.news.naver.com/article/009/0005595470?ntype=RANKING \
    https://n.news.naver.com/article/023/0003942879?ntype=RANKING \
    https://n.news.naver.com/article/018/0006170637?ntype=RANKING \
    https://n.news.naver.com/article/016/0002562292?ntype=RANKING \
    https://n.news.naver.com/article/025/0003484923?ntype=RANKING \
    https://n.news.naver.com/article/025/0003484407?ntype=RANKING \
    https://n.news.naver.com/article/025/0003482886?ntype=RANKING
```

## 📋 수집된 데이터 확인

수집이 완료되면 다음 위치에 CSV 파일이 생성됩니다:

```
data/raw/
├── comments_politics.csv
├── comments_society.csv
└── comments_entertainment.csv
```

데이터 확인:
```bash
ls -lh data/raw/comments_*.csv
```

## ⚙️ 옵션 설명

- `--section`: 섹션 이름 (politics/society/entertainment)
- `--target_count`: 목표 댓글 개수 (기본값: 500)
- `--max_clicks`: 기사당 최대 "더보기" 클릭 횟수 (기본값: 30)
- `--headless`: 헤드리스 모드 (브라우저 창 숨김)
- `--no-headless`: 헤드리스 모드 비활성화 (브라우저 창 표시, 디버깅용)
- `--urls`: 기사 URL 리스트 (공백으로 구분)

## 🔍 수집 과정

1. 각 섹션의 기사 URL을 순차적으로 처리
2. 각 기사에서 댓글 수집
3. 목표 개수(500개)에 도달하면 자동 중단
4. 중복 댓글 자동 제거
5. 섹션별 CSV 파일로 저장

## ⚠️ 주의사항

- 네이버 뉴스의 댓글 시스템이 변경되면 셀렉터를 수정해야 할 수 있습니다
- 수집 속도가 너무 빠르면 IP 차단될 수 있으니 적절한 대기 시간이 포함되어 있습니다
- 헤드리스 모드에서 문제가 발생하면 `--no-headless` 옵션으로 브라우저 창을 확인하세요

## 🐛 문제 해결

### ChromeDriver 오류
```bash
# ChromeDriver 재설치
pip install --upgrade webdriver-manager
```

### 댓글이 수집되지 않는 경우
1. `--no-headless` 옵션으로 브라우저 창을 확인
2. 네이버 뉴스 페이지 구조가 변경되었는지 확인
3. `src/scrape_comments.py`의 셀렉터 확인 및 수정

### 목표 개수에 도달하지 못한 경우
- `--max_clicks` 값을 증가시키기 (예: 50)
- 더 많은 기사 URL 추가

