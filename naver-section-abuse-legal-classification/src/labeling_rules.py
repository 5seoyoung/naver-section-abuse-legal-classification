"""
라벨링 규칙 및 유틸리티 함수
"""
import re
from typing import List, Dict, Optional


# 라벨 정의
LABELS = {
    'A': '명예훼손형',
    'B': '모욕형',
    'C': '혐오표현형',
    'D': '협박·위협형',
    'E': '성희롱·성폭력형',
    'F': '악플 아님'
}

# 라벨 우선순위 (높을수록 우선)
LABEL_PRIORITY = {
    'D': 6,  # 협박·위협형 (가장 심각)
    'E': 5,  # 성희롱·성폭력형
    'A': 4,  # 명예훼손형
    'C': 3,  # 혐오표현형
    'B': 2,  # 모욕형
    'F': 1   # 악플 아님
}


def get_label_name(label: str) -> str:
    """라벨 코드를 한글 이름으로 변환"""
    return LABELS.get(label, 'Unknown')


def get_label_priority(label: str) -> int:
    """라벨의 우선순위 반환"""
    return LABEL_PRIORITY.get(label, 0)


def select_primary_label(labels: List[str]) -> str:
    """
    여러 라벨 중 가장 우선순위가 높은 라벨 선택
    
    Args:
        labels: 라벨 리스트
        
    Returns:
        우선순위가 가장 높은 라벨
    """
    if not labels:
        return 'F'
    
    return max(labels, key=get_label_priority)


def anonymize_text(text: str, person_patterns: List[str] = None, 
                   group_patterns: List[str] = None) -> str:
    """
    텍스트에서 인물명과 집단명을 토큰으로 대체
    
    Args:
        text: 원본 텍스트
        person_patterns: 인물명 패턴 리스트
        group_patterns: 집단명 패턴 리스트
        
    Returns:
        익명화된 텍스트
    """
    anonymized = text
    
    if person_patterns:
        for pattern in person_patterns:
            anonymized = re.sub(pattern, '[인물]', anonymized, flags=re.IGNORECASE)
    
    if group_patterns:
        for pattern in group_patterns:
            anonymized = re.sub(pattern, '[집단]', anonymized, flags=re.IGNORECASE)
    
    return anonymized


def validate_label(label: str) -> bool:
    """라벨이 유효한지 검증"""
    return label in LABELS


def get_all_labels() -> List[str]:
    """모든 라벨 리스트 반환"""
    return list(LABELS.keys())


def get_label_distribution(labels: List[str]) -> Dict[str, int]:
    """라벨 분포 계산"""
    distribution = {label: 0 for label in LABELS.keys()}
    for label in labels:
        if label in distribution:
            distribution[label] += 1
    return distribution

