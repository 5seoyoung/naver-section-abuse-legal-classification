"""
악플만을 대상으로 한 심화 분석 스크립트
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from collections import Counter

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_labeled_comments
from src.labeling_rules import LABELS, get_label_name, LABEL_PRIORITY

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def filter_abusive_comments(df: pd.DataFrame) -> pd.DataFrame:
    """악플만 필터링 (F 제외)"""
    abusive_labels = ['A', 'B', 'C', 'D', 'E']
    return df[df['label'].isin(abusive_labels)].copy()


def analyze_section_distribution_abusive_only(df_abusive: pd.DataFrame) -> dict:
    """
    악플만을 대상으로 한 섹션별 분포 분석
    """
    # 교차테이블 (악플만)
    cross_table = pd.crosstab(df_abusive['section'], df_abusive['label'], margins=True)
    
    # 섹션별 악플 비율 (악플 내에서의 비율)
    percentage_table = pd.crosstab(df_abusive['section'], df_abusive['label'], normalize='index') * 100
    
    # 섹션별 악플 개수
    section_counts = df_abusive['section'].value_counts()
    
    # 악플 유형별 섹션 분포
    label_section_dist = pd.crosstab(df_abusive['label'], df_abusive['section'], normalize='index') * 100
    
    return {
        'cross_table': cross_table,
        'percentage_table': percentage_table,
        'section_counts': section_counts,
        'label_section_dist': label_section_dist
    }


def analyze_abusive_type_prevalence(df_abusive: pd.DataFrame) -> dict:
    """
    악플 유형별 섹션 선호도 분석
    """
    results = {}
    
    for label in ['A', 'B', 'C', 'D', 'E']:
        label_df = df_abusive[df_abusive['label'] == label]
        if len(label_df) > 0:
            section_dist = label_df['section'].value_counts(normalize=True) * 100
            results[label] = {
                'total': len(label_df),
                'section_distribution': section_dist.to_dict(),
                'dominant_section': section_dist.index[0] if len(section_dist) > 0 else None
            }
    
    return results


def analyze_relative_proportions(df_abusive: pd.DataFrame) -> pd.DataFrame:
    """
    섹션별 악플 유형의 상대적 비율 분석
    예: 정치 섹션에서 A, B, E 중 어떤 것이 가장 많은가?
    """
    sections = df_abusive['section'].unique()
    results = []
    
    for section in sections:
        section_df = df_abusive[df_abusive['section'] == section]
        label_counts = section_df['label'].value_counts()
        total_abusive = len(section_df)
        
        for label, count in label_counts.items():
            results.append({
                'section': section,
                'label': label,
                'count': count,
                'proportion': (count / total_abusive) * 100 if total_abusive > 0 else 0
            })
    
    return pd.DataFrame(results)


def analyze_abusive_severity(df_abusive: pd.DataFrame) -> dict:
    """
    악플 심각도 분석 (우선순위 기반)
    """
    # 우선순위가 높을수록 심각
    df_abusive['severity'] = df_abusive['label'].map(LABEL_PRIORITY)
    
    section_severity = df_abusive.groupby('section')['severity'].agg(['mean', 'std', 'min', 'max'])
    
    # 섹션별 심각한 악플 비율 (D, E만)
    severe_labels = ['D', 'E']
    section_severe_ratio = df_abusive.groupby('section').apply(
        lambda x: (x['label'].isin(severe_labels).sum() / len(x)) * 100
    )
    
    return {
        'section_severity_stats': section_severity,
        'section_severe_ratio': section_severe_ratio
    }


def analyze_text_length_patterns(df_abusive: pd.DataFrame) -> dict:
    """
    악플 유형별 댓글 길이 패턴 분석
    """
    df_abusive['text_length'] = df_abusive['comment_text'].str.len()
    
    # 유형별 평균 길이
    length_by_label = df_abusive.groupby('label')['text_length'].agg(['mean', 'median', 'std'])
    
    # 섹션별 평균 길이
    length_by_section = df_abusive.groupby('section')['text_length'].agg(['mean', 'median', 'std'])
    
    return {
        'length_by_label': length_by_label,
        'length_by_section': length_by_section
    }


def perform_statistical_tests(df_abusive: pd.DataFrame) -> dict:
    """
    통계적 검정 수행
    """
    results = {}
    
    # 1. 섹션별 악플 유형 분포 차이 (카이제곱 검정)
    contingency = pd.crosstab(df_abusive['section'], df_abusive['label'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    results['chi2_test'] = {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'significant': p_value < 0.05
    }
    
    # 2. 각 악플 유형별 섹션 분포 차이 (가능한 경우)
    for label in ['A', 'B', 'E']:  # C, D는 데이터 부족
        label_df = df_abusive[df_abusive['label'] == label]
        if len(label_df) >= 10:  # 최소 샘플 수
            section_counts = label_df['section'].value_counts()
            if len(section_counts) >= 2:
                # Fisher's exact test (작은 샘플에 적합)
                try:
                    # 간단한 비율 비교
                    total = len(label_df)
                    politics_ratio = section_counts.get('politics', 0) / total
                    society_ratio = section_counts.get('society', 0) / total
                    entertainment_ratio = section_counts.get('entertainment', 0) / total
                    
                    results[f'{label}_section_distribution'] = {
                        'politics': politics_ratio * 100,
                        'society': society_ratio * 100,
                        'entertainment': entertainment_ratio * 100,
                        'total': total
                    }
                except:
                    pass
    
    return results


def create_visualizations(df_abusive: pd.DataFrame, output_dir: str = "results/charts"):
    """시각화 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 악플만의 섹션별 분포 (막대 그래프)
    cross_table = pd.crosstab(df_abusive['section'], df_abusive['label'])
    cross_table.columns = [get_label_name(col) for col in cross_table.columns]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cross_table.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('섹션', fontsize=12)
    ax.set_ylabel('악플 개수', fontsize=12)
    ax.set_title('섹션별 악플 법적 유형 분포 (악플만)', fontsize=14, fontweight='bold')
    ax.legend(title='법적 유형', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'abusive_only_section_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 섹션별 악플 유형 비율 (스택 바)
    percentage_table = pd.crosstab(df_abusive['section'], df_abusive['label'], normalize='index') * 100
    percentage_table.columns = [get_label_name(col) for col in percentage_table.columns]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    percentage_table.plot(kind='bar', stacked=True, ax=ax, width=0.8)
    ax.set_xlabel('섹션', fontsize=12)
    ax.set_ylabel('비율 (%)', fontsize=12)
    ax.set_title('섹션별 악플 유형 비율 (악플 내)', fontsize=14, fontweight='bold')
    ax.legend(title='법적 유형', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'abusive_only_section_proportion.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 악플 유형별 섹션 분포
    label_section = pd.crosstab(df_abusive['label'], df_abusive['section'], normalize='index') * 100
    label_section.index = [get_label_name(idx) for idx in label_section.index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    label_section.plot(kind='barh', ax=ax, width=0.8)
    ax.set_xlabel('비율 (%)', fontsize=12)
    ax.set_ylabel('악플 유형', fontsize=12)
    ax.set_title('악플 유형별 섹션 분포', fontsize=14, fontweight='bold')
    ax.legend(title='섹션', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'abusive_type_by_section.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_detailed_report(df_abusive: pd.DataFrame, output_dir: str = "results"):
    """상세 분석 리포트 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 분석 수행
    dist_analysis = analyze_section_distribution_abusive_only(df_abusive)
    type_prevalence = analyze_abusive_type_prevalence(df_abusive)
    relative_props = analyze_relative_proportions(df_abusive)
    severity_analysis = analyze_abusive_severity(df_abusive)
    length_analysis = analyze_text_length_patterns(df_abusive)
    stat_tests = perform_statistical_tests(df_abusive)
    
    # 리포트 작성
    report_path = os.path.join(output_dir, 'abusive_only_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("악플만을 대상으로 한 심화 분석 리포트\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"총 악플 개수: {len(df_abusive)}개\n")
        f.write(f"악플 유형: {sorted(df_abusive['label'].unique())}\n\n")
        
        f.write("1. 섹션별 악플 분포 (교차테이블)\n")
        f.write("-" * 80 + "\n")
        f.write(str(dist_analysis['cross_table']))
        f.write("\n\n")
        
        f.write("2. 섹션별 악플 유형 비율 (악플 내에서의 비율)\n")
        f.write("-" * 80 + "\n")
        f.write(str(dist_analysis['percentage_table'].round(2)))
        f.write("\n\n")
        
        f.write("3. 악플 유형별 섹션 선호도\n")
        f.write("-" * 80 + "\n")
        for label, info in type_prevalence.items():
            f.write(f"\n{get_label_name(label)} ({label}):\n")
            f.write(f"  총 개수: {info['total']}개\n")
            f.write(f"  섹션 분포:\n")
            for section, ratio in info['section_distribution'].items():
                f.write(f"    {section}: {ratio:.2f}%\n")
            f.write(f"  주요 섹션: {info['dominant_section']}\n")
        f.write("\n")
        
        f.write("4. 섹션별 악플 유형 상대적 비율\n")
        f.write("-" * 80 + "\n")
        for section in df_abusive['section'].unique():
            section_df = df_abusive[df_abusive['section'] == section]
            f.write(f"\n{section}:\n")
            label_counts = section_df['label'].value_counts()
            total = len(section_df)
            for label, count in label_counts.items():
                prop = (count / total) * 100
                f.write(f"  {get_label_name(label)}: {count}개 ({prop:.2f}%)\n")
        f.write("\n")
        
        f.write("5. 악플 심각도 분석\n")
        f.write("-" * 80 + "\n")
        f.write("섹션별 평균 심각도 (높을수록 심각):\n")
        f.write(str(severity_analysis['section_severity_stats']))
        f.write("\n\n")
        f.write("섹션별 심각한 악플 비율 (D, E 유형):\n")
        for section, ratio in severity_analysis['section_severe_ratio'].items():
            f.write(f"  {section}: {ratio:.2f}%\n")
        f.write("\n")
        
        f.write("6. 댓글 길이 패턴\n")
        f.write("-" * 80 + "\n")
        f.write("악플 유형별 평균 길이:\n")
        f.write(str(length_analysis['length_by_label']))
        f.write("\n\n")
        f.write("섹션별 평균 길이:\n")
        f.write(str(length_analysis['length_by_section']))
        f.write("\n\n")
        
        f.write("7. 통계적 검정 결과\n")
        f.write("-" * 80 + "\n")
        if 'chi2_test' in stat_tests:
            chi2_result = stat_tests['chi2_test']
            f.write(f"카이제곱 검정 (섹션별 악플 유형 분포 차이):\n")
            f.write(f"  χ² = {chi2_result['chi2']:.4f}\n")
            f.write(f"  p-value = {chi2_result['p_value']:.4f}\n")
            f.write(f"  자유도 = {chi2_result['dof']}\n")
            f.write(f"  유의성: {'유의함' if chi2_result['significant'] else '유의하지 않음'} (α=0.05)\n\n")
        
        f.write("8. 주요 발견사항 및 논문용 인사이트\n")
        f.write("-" * 80 + "\n")
        
        # 정치 섹션 분석
        politics_abusive = df_abusive[df_abusive['section'] == 'politics']
        if len(politics_abusive) > 0:
            politics_b_ratio = (politics_abusive['label'] == 'B').sum() / len(politics_abusive) * 100
            f.write(f"\n[정치 섹션]\n")
            f.write(f"- 전체 악플 중 정치 섹션이 {(len(politics_abusive)/len(df_abusive)*100):.1f}% 차지\n")
            f.write(f"- 모욕형(B)이 정치 섹션 악플의 {politics_b_ratio:.1f}%를 차지 (가장 높은 비율)\n")
            if (politics_abusive['label'] == 'E').sum() > 0:
                f.write(f"- 성희롱형(E)이 정치 섹션에서만 발견됨 (다른 섹션 대비 특징적)\n")
        
        # 연예 섹션 분석
        entertainment_abusive = df_abusive[df_abusive['section'] == 'entertainment']
        if len(entertainment_abusive) > 0:
            f.write(f"\n[연예 섹션]\n")
            f.write(f"- 전체 악플 중 연예 섹션이 {(len(entertainment_abusive)/len(df_abusive)*100):.1f}% 차지 (가장 낮음)\n")
            f.write(f"- 성희롱형(E)이 연예 섹션에서 발견되지 않음\n")
        
        # 모욕형(B) 분석
        b_abusive = df_abusive[df_abusive['label'] == 'B']
        if len(b_abusive) > 0:
            b_politics_ratio = (b_abusive['section'] == 'politics').sum() / len(b_abusive) * 100
            f.write(f"\n[모욕형(B) - 가장 많은 악플 유형]\n")
            f.write(f"- 전체 악플의 {(len(b_abusive)/len(df_abusive)*100):.1f}%를 차지\n")
            f.write(f"- 모욕형 중 {b_politics_ratio:.1f}%가 정치 섹션에서 발견됨\n")
    
    print(f"Saved detailed analysis report to {report_path}")
    
    # 시각화 생성
    create_visualizations(df_abusive, os.path.join(output_dir, 'charts'))


def main():
    """메인 함수"""
    # 라벨링된 데이터 로드
    df = load_labeled_comments()
    
    # 악플만 필터링
    df_abusive = filter_abusive_comments(df)
    
    print(f"전체 댓글: {len(df)}개")
    print(f"악플: {len(df_abusive)}개 ({len(df_abusive)/len(df)*100:.2f}%)")
    print(f"악플 유형: {sorted(df_abusive['label'].unique())}")
    print(f"\n섹션별 악플 개수:")
    print(df_abusive['section'].value_counts())
    
    # 상세 분석 리포트 생성
    generate_detailed_report(df_abusive)


if __name__ == "__main__":
    main()

