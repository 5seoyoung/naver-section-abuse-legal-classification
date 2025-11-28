"""
여러 기사에서 댓글을 배치로 수집하는 스크립트
섹션별로 목표 개수만큼 수집
"""
import os
import sys
import time
import argparse
import pandas as pd

# 프로젝트 루트를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 직접 import
try:
    from src.scrape_comments import scrape_naver_comments
except ImportError:
    # 대체 방법: 상대 import
    from .scrape_comments import scrape_naver_comments


def collect_comments_by_section(urls: list, section: str, target_count: int = 500,
                                max_clicks_per_article: int = 30, headless: bool = True):
    """
    섹션별로 여러 기사에서 댓글을 수집하여 목표 개수만큼 모음
    
    Args:
        urls: 기사 URL 리스트
        section: 섹션 이름
        target_count: 목표 댓글 개수
        max_clicks_per_article: 기사당 최대 '더보기' 클릭 횟수
        headless: 헤드리스 모드 여부
        
    Returns:
        수집된 댓글 리스트
    """
    all_comments = []
    collected_count = 0
    
    print(f"\n{'='*60}")
    print(f"섹션: {section}")
    print(f"목표 개수: {target_count}개")
    print(f"기사 수: {len(urls)}개")
    print(f"{'='*60}\n")
    
    for idx, url in enumerate(urls, 1):
        if collected_count >= target_count:
            print(f"\n목표 개수({target_count}개) 달성! 수집 중단.")
            break
        
        remaining = target_count - collected_count
        print(f"\n[{idx}/{len(urls)}] 기사 처리 중...")
        print(f"URL: {url}")
        print(f"현재 수집: {collected_count}개 / 목표: {target_count}개 (남은 개수: {remaining}개)")
        
        try:
            # 댓글 수집
            comments = scrape_naver_comments(
                article_url=url,
                max_clicks=max_clicks_per_article,
                headless=headless
            )
            
            # 중복 제거 (같은 댓글이 여러 번 수집될 수 있음)
            new_comments = [c for c in comments if c not in all_comments]
            
            # 목표 개수까지만 추가
            needed = target_count - collected_count
            if len(new_comments) > needed:
                new_comments = new_comments[:needed]
            
            all_comments.extend(new_comments)
            collected_count = len(all_comments)
            
            print(f"이번 기사에서 수집: {len(new_comments)}개")
            print(f"누적 수집: {collected_count}개")
            
            # API 호출 제한을 위한 대기
            time.sleep(2)
            
        except Exception as e:
            print(f"오류 발생: {e}")
            print(f"다음 기사로 넘어갑니다...")
            continue
    
    print(f"\n{'='*60}")
    print(f"섹션 '{section}' 수집 완료!")
    print(f"총 수집 개수: {len(all_comments)}개")
    print(f"{'='*60}\n")
    
    return all_comments


def save_comments_to_csv(comments, section: str, out_dir: str = "data/raw"):
    """
    수집된 댓글을 CSV로 저장
    
    Args:
        comments: 댓글 리스트
        section: 섹션 이름
        out_dir: 출력 디렉토리
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(
        {
            "id": [f"{section}_{i+1:04d}" for i in range(len(comments))],
            "section": section,
            "comment_text": comments,
        }
    )
    out_path = os.path.join(out_dir, f"comments_{section}.csv")
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"✅ 저장 완료: {out_path} ({len(df)}개 댓글)")


def main():
    parser = argparse.ArgumentParser(description='네이버 뉴스 댓글 배치 수집')
    parser.add_argument(
        '--section',
        type=str,
        required=True,
        choices=['politics', 'society', 'entertainment'],
        help='뉴스 섹션 (politics/society/entertainment)'
    )
    parser.add_argument(
        '--urls',
        type=str,
        nargs='+',
        required=True,
        help='기사 URL 리스트 (공백으로 구분)'
    )
    parser.add_argument(
        '--target_count',
        type=int,
        default=500,
        help='섹션별 목표 댓글 개수 (기본값: 500)'
    )
    parser.add_argument(
        '--max_clicks',
        type=int,
        default=30,
        help='기사당 최대 "더보기" 클릭 횟수 (기본값: 30)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='헤드리스 모드로 실행'
    )
    parser.add_argument(
        '--no-headless',
        dest='headless',
        action='store_false',
        help='헤드리스 모드 비활성화 (브라우저 창 표시)'
    )
    parser.set_defaults(headless=True)
    
    args = parser.parse_args()
    
    # 댓글 수집
    comments = collect_comments_by_section(
        urls=args.urls,
        section=args.section,
        target_count=args.target_count,
        max_clicks_per_article=args.max_clicks,
        headless=args.headless
    )
    
    # CSV로 저장
    save_comments_to_csv(comments, section=args.section)


if __name__ == "__main__":
    main()

