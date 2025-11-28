"""
섹션별 악플 분포 분석 스크립트
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_labeled_comments
from src.labeling_rules import LABELS, get_label_name

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False


def create_cross_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    섹션 × 라벨 교차테이블 생성
    
    Args:
        df: 라벨링된 데이터프레임
        
    Returns:
        교차테이블 (섹션 × 라벨)
    """
    cross_table = pd.crosstab(df['section'], df['label'], margins=True)
    return cross_table


def create_percentage_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    섹션별 라벨 비율 테이블 생성 (백분율)
    
    Args:
        df: 라벨링된 데이터프레임
        
    Returns:
        섹션별 라벨 비율 테이블
    """
    cross_table = pd.crosstab(df['section'], df['label'], normalize='index') * 100
    return cross_table.round(2)


def get_top_labels_by_section(df: pd.DataFrame, top_n: int = 2) -> dict:
    """
    섹션별 가장 많이 등장한 라벨 Top N 추출
    
    Args:
        df: 라벨링된 데이터프레임
        top_n: 상위 N개
        
    Returns:
        섹션별 Top 라벨 딕셔너리
    """
    sections = df['section'].unique()
    result = {}
    
    for section in sections:
        section_df = df[df['section'] == section]
        label_counts = section_df['label'].value_counts()
        top_labels = label_counts.head(top_n).to_dict()
        result[section] = top_labels
    
    return result


def chi_square_test(df: pd.DataFrame) -> dict:
    """
    카이제곱 검정 수행 (섹션별 악플 분포 차이 검정)
    
    Args:
        df: 라벨링된 데이터프레임
        
    Returns:
        검정 결과 딕셔너리
    """
    contingency_table = pd.crosstab(df['section'], df['label'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'significant': p_value < 0.05
    }


def plot_section_label_distribution(df: pd.DataFrame, 
                                    output_path: str = "results/charts/section_label_distribution.png"):
    """
    섹션별 라벨 분포 막대 그래프 생성
    
    Args:
        df: 라벨링된 데이터프레임
        output_path: 저장 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 교차테이블 생성 (비율)
    percentage_table = create_percentage_table(df)
    
    # 한글 라벨명으로 변환
    percentage_table.columns = [get_label_name(col) for col in percentage_table.columns]
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    percentage_table.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('섹션', fontsize=12)
    ax.set_ylabel('비율 (%)', fontsize=12)
    ax.set_title('섹션별 악플 법적 유형 분포', fontsize=14, fontweight='bold')
    ax.legend(title='법적 유형', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution chart to {output_path}")
    plt.close()


def plot_label_counts_by_section(df: pd.DataFrame,
                                 output_path: str = "results/charts/label_counts_by_section.png"):
    """
    섹션별 라벨 개수 히트맵 생성
    
    Args:
        df: 라벨링된 데이터프레임
        output_path: 저장 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 교차테이블 생성
    cross_table = pd.crosstab(df['section'], df['label'])
    
    # 한글 라벨명으로 변환
    cross_table.columns = [get_label_name(col) for col in cross_table.columns]
    
    # 히트맵 생성
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_table, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': '개수'})
    plt.title('섹션별 악플 법적 유형 개수', fontsize=14, fontweight='bold')
    plt.xlabel('법적 유형', fontsize=12)
    plt.ylabel('섹션', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def generate_analysis_report(df: pd.DataFrame, output_dir: str = "results"):
    """
    전체 분석 리포트 생성
    
    Args:
        df: 라벨링된 데이터프레임
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 교차테이블 저장
    cross_table = create_cross_table(df)
    cross_table_path = os.path.join(output_dir, "cross_table.csv")
    cross_table.to_csv(cross_table_path, encoding='utf-8-sig')
    print(f"\n=== 교차테이블 ===")
    print(cross_table)
    print(f"\nSaved to {cross_table_path}")
    
    # 2. 비율 테이블 저장
    percentage_table = create_percentage_table(df)
    percentage_table_path = os.path.join(output_dir, "percentage_table.csv")
    percentage_table.to_csv(percentage_table_path, encoding='utf-8-sig')
    print(f"\n=== 섹션별 비율 테이블 (%) ===")
    print(percentage_table)
    print(f"\nSaved to {percentage_table_path}")
    
    # 3. Top 라벨 추출
    top_labels = get_top_labels_by_section(df, top_n=2)
    print(f"\n=== 섹션별 Top 2 라벨 ===")
    for section, labels in top_labels.items():
        print(f"\n{section}:")
        for label, count in labels.items():
            print(f"  {get_label_name(label)}: {count}개")
    
    # 4. 카이제곱 검정
    chi2_result = chi_square_test(df)
    print(f"\n=== 카이제곱 검정 결과 ===")
    print(f"카이제곱 통계량: {chi2_result['chi2_statistic']:.4f}")
    print(f"p-value: {chi2_result['p_value']:.4f}")
    print(f"자유도: {chi2_result['degrees_of_freedom']}")
    print(f"통계적 유의성: {'유의함' if chi2_result['significant'] else '유의하지 않음'} (α=0.05)")
    
    # 5. 그래프 생성
    plot_section_label_distribution(df, 
                                   os.path.join(output_dir, "charts/section_label_distribution.png"))
    plot_label_counts_by_section(df, 
                                 os.path.join(output_dir, "charts/label_counts_by_section.png"))
    
    # 6. 리포트 텍스트 저장
    report_path = os.path.join(output_dir, "distribution_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 섹션별 악플 분포 분석 리포트 ===\n\n")
        f.write("1. 교차테이블\n")
        f.write(str(cross_table))
        f.write("\n\n2. 섹션별 비율 테이블 (%)\n")
        f.write(str(percentage_table))
        f.write("\n\n3. 섹션별 Top 2 라벨\n")
        for section, labels in top_labels.items():
            f.write(f"\n{section}:\n")
            for label, count in labels.items():
                f.write(f"  {get_label_name(label)}: {count}개\n")
        f.write("\n4. 카이제곱 검정 결과\n")
        f.write(f"카이제곱 통계량: {chi2_result['chi2_statistic']:.4f}\n")
        f.write(f"p-value: {chi2_result['p_value']:.4f}\n")
        f.write(f"자유도: {chi2_result['degrees_of_freedom']}\n")
        f.write(f"통계적 유의성: {'유의함' if chi2_result['significant'] else '유의하지 않음'} (α=0.05)\n")
    
    print(f"\nSaved analysis report to {report_path}")


def main():
    """메인 함수"""
    # 라벨링된 데이터 로드
    df = load_labeled_comments()
    
    print(f"Loaded {len(df)} labeled comments")
    print(f"Sections: {df['section'].unique()}")
    print(f"Labels: {df['label'].unique()}")
    
    # 분석 리포트 생성
    generate_analysis_report(df)


if __name__ == "__main__":
    main()

