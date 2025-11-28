"""
텍스트 전처리 모듈
"""
import re
from typing import List, Optional


def normalize_emoticons(text: str) -> str:
    """
    이모티콘 및 반복 문자 정규화
    
    Args:
        text: 원본 텍스트
        
    Returns:
        정규화된 텍스트
    """
    # 반복 문자 정규화 (예: ㅋㅋㅋ -> ㅋㅋ, ㅠㅠㅠ -> ㅠㅠ)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 이모티콘 제거 (선택적)
    # text = re.sub(r'[^\w\s가-힣]', '', text)  # 필요시 주석 해제
    
    return text


def replace_vulgar_words(text: str, replacement_dict: Optional[dict] = None) -> str:
    """
    욕설 변형 치환 (예: ㅂㅅ -> 바보)
    
    Args:
        text: 원본 텍스트
        replacement_dict: 치환 사전 (None이면 기본 사전 사용)
        
    Returns:
        치환된 텍스트
    """
    if replacement_dict is None:
        # 기본 치환 사전 (예시)
        replacement_dict = {
            'ㅂㅅ': '바보',
            'ㅅㅂ': '시발',
            'ㅈㄴ': '정말',
            # 실제 프로젝트에서는 더 많은 패턴 필요
        }
    
    for pattern, replacement in replacement_dict.items():
        text = text.replace(pattern, replacement)
    
    return text


def apply_tokenization(text: str, person_token: str = '[인물]', 
                      group_token: str = '[집단]') -> str:
    """
    인물명과 집단명을 토큰으로 대체
    
    Args:
        text: 원본 텍스트
        person_token: 인물 토큰
        group_token: 집단 토큰
        
    Returns:
        토큰화된 텍스트
    """
    # 실제 구현에서는 NER 모델이나 사전을 사용하여 인물/집단명 추출
    # 여기서는 기본적인 패턴 매칭만 수행
    # 실제 프로젝트에서는 KoNLPy나 다른 NER 도구 사용 권장
    
    return text


def preprocess_text(text: str, normalize: bool = True, 
                   replace_vulgar: bool = True, 
                   tokenize: bool = True) -> str:
    """
    전체 전처리 파이프라인
    
    Args:
        text: 원본 텍스트
        normalize: 이모티콘/반복 문자 정규화 여부
        replace_vulgar: 욕설 치환 여부
        tokenize: 토큰화 여부
        
    Returns:
        전처리된 텍스트
    """
    processed = text.strip()
    
    if normalize:
        processed = normalize_emoticons(processed)
    
    if replace_vulgar:
        processed = replace_vulgar_words(processed)
    
    if tokenize:
        processed = apply_tokenization(processed)
    
    return processed


def preprocess_batch(texts: List[str], **kwargs) -> List[str]:
    """
    여러 텍스트를 일괄 전처리
    
    Args:
        texts: 텍스트 리스트
        **kwargs: preprocess_text에 전달할 인자
        
    Returns:
        전처리된 텍스트 리스트
    """
    return [preprocess_text(text, **kwargs) for text in texts]

