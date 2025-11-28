"""
특징 추출 모듈 (TF-IDF, KoBERT 임베딩)
"""
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel


class TfidfFeatureExtractor:
    """TF-IDF 특징 추출기"""
    
    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), 
                 max_features: int = 5000, min_df: int = 2):
        """
        Args:
            ngram_range: n-gram 범위
            max_features: 최대 특징 수
            min_df: 최소 문서 빈도
        """
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            analyzer='char'  # 한글 문자 단위 분석
        )
        self.fitted = False
    
    def fit(self, texts: List[str]):
        """TF-IDF 벡터화기 학습"""
        self.vectorizer.fit(texts)
        self.fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """텍스트를 TF-IDF 벡터로 변환"""
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """학습 및 변환 동시 수행"""
        self.fit(texts)
        return self.transform(texts)


class KoBERTFeatureExtractor:
    """KoBERT 임베딩 특징 추출기"""
    
    def __init__(self, model_name: str = "monologg/kobert", 
                 device: str = None, max_length: int = 128):
        """
        Args:
            model_name: KoBERT 모델 이름
            device: 사용할 디바이스 ('cuda' or 'cpu')
            max_length: 최대 시퀀스 길이
        """
        self.model_name = model_name
        self.max_length = max_length
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 모델 및 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_cls_embedding(self, texts: List[str]) -> np.ndarray:
        """
        [CLS] 토큰의 임베딩 벡터 추출
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            [CLS] 임베딩 벡터 배열 (n_samples, hidden_size)
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # 토크나이징
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # 디바이스로 이동
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # 모델 통과
                outputs = self.model(**encoded)
                
                # [CLS] 토큰 임베딩 추출 (첫 번째 토큰)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])
        
        return np.array(embeddings)
    
    def extract_mean_pooling(self, texts: List[str]) -> np.ndarray:
        """
        평균 풀링 임베딩 추출
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            평균 풀링 임베딩 벡터 배열
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                
                # 평균 풀링 (attention mask 고려)
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                # attention mask로 실제 토큰만 평균 계산
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_embedding = (sum_embeddings / sum_mask).cpu().numpy()
                
                embeddings.append(mean_embedding[0])
        
        return np.array(embeddings)


def get_feature_extractor(method: str = 'tfidf', **kwargs):
    """
    특징 추출기 팩토리 함수
    
    Args:
        method: 추출 방법 ('tfidf' or 'kobert')
        **kwargs: 특징 추출기에 전달할 인자
        
    Returns:
        특징 추출기 인스턴스
    """
    if method == 'tfidf':
        return TfidfFeatureExtractor(**kwargs)
    elif method == 'kobert':
        return KoBERTFeatureExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'tfidf' or 'kobert'")

