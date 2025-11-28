#!/bin/bash

# 모든 섹션의 댓글을 수집하는 스크립트

# 가상환경 활성화
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "가상환경 활성화됨"
fi

echo "=========================================="
echo "네이버 뉴스 댓글 수집 시작"
echo "섹션별 목표: 500개씩 (총 1500개)"
echo "=========================================="

# 정치 섹션
echo ""
echo ">>> 정치 섹션 수집 시작..."
python3 -m src.scrape_batch \
    --section politics \
    --target_count 500 \
    --max_clicks 30 \
    --headless \
    --urls \
    https://n.news.naver.com/article/052/0002277256 \
    https://n.news.naver.com/article/011/0004559400?ntype=RANKING \
    https://n.news.naver.com/article/052/0002277544?ntype=RANKING

# 사회 섹션
echo ""
echo ">>> 사회 섹션 수집 시작..."
python3 -m src.scrape_batch \
    --section society \
    --target_count 500 \
    --max_clicks 30 \
    --headless \
    --urls \
    https://n.news.naver.com/article/011/0004559349?ntype=RANKING \
    https://n.news.naver.com/article/079/0004089120?ntype=RANKING \
    https://n.news.naver.com/article/081/0003594894?ntype=RANKING

# 연예 섹션
echo ""
echo ">>> 연예 섹션 수집 시작..."
python3 -m src.scrape_batch \
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

echo ""
echo "=========================================="
echo "모든 섹션 수집 완료!"
echo "=========================================="
echo ""
echo "수집된 데이터 확인:"
ls -lh data/raw/comments_*.csv

