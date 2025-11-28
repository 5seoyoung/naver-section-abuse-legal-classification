"""
데이터 로더 모듈
"""
import os
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


def load_raw_comments(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    섹션별 원본 댓글 데이터 로드
    
    Args:
        data_dir: 원본 데이터 디렉토리
        
    Returns:
        모든 섹션의 댓글을 합친 DataFrame
    """
    sections = ["politics", "society", "entertainment"]
    dfs = []
    
    for section in sections:
        file_path = os.path.join(data_dir, f"comments_{section}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"Warning: {file_path} not found")
    
    if not dfs:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def load_labeled_comments(data_path: str = "data/processed/comments_labeled.csv") -> pd.DataFrame:
    """
    라벨링된 댓글 데이터 로드
    
    Args:
        data_path: 라벨링된 데이터 파일 경로
        
    Returns:
        라벨링된 댓글 DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Labeled data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # 필수 컬럼 확인
    required_cols = ['id', 'section', 'comment_text', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, 
               val_size: float = 0.2, random_state: int = 42,
               stratify_col: Optional[str] = 'label') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    데이터를 train/val/test로 분할 (Stratified Split)
    
    Args:
        df: 전체 데이터프레임
        test_size: 테스트 세트 비율
        val_size: 검증 세트 비율 (train에서 분할)
        random_state: 랜덤 시드
        stratify_col: 계층화 기준 컬럼
        
    Returns:
        (train_df, val_df, test_df)
    """
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    
    # 먼저 train+val과 test로 분할
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    # train_val에서 train과 val로 분할
    if stratify_col and stratify_col in train_val_df.columns:
        stratify = train_val_df[stratify_col]
    else:
        stratify = None
    
    val_size_adjusted = val_size / (1 - test_size)  # 전체 대비 비율로 조정
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify
    )
    
    return train_df, val_df, test_df


def get_section_data(df: pd.DataFrame, section: str) -> pd.DataFrame:
    """
    특정 섹션의 데이터만 추출
    
    Args:
        df: 전체 데이터프레임
        section: 섹션 이름 ('politics', 'society', 'entertainment')
        
    Returns:
        해당 섹션의 데이터프레임
    """
    return df[df['section'] == section].copy()


def save_labeled_data(df: pd.DataFrame, output_path: str = "data/processed/comments_labeled.csv"):
    """
    라벨링된 데이터를 CSV로 저장
    
    Args:
        df: 라벨링된 데이터프레임
        output_path: 저장 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved labeled data to {output_path} ({len(df)} rows)")

