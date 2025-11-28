"""
모델 학습 스크립트 (3가지 baseline 모델)
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from tqdm import tqdm

import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_labeled_comments, split_data
from src.preprocessing import preprocess_batch
from src.feature_extraction import TfidfFeatureExtractor, KoBERTFeatureExtractor
from src.utils import save_results_summary, ensure_dir


class CommentDataset(Dataset):
    """댓글 데이터셋"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class KoBERTClassifier(nn.Module):
    """KoBERT 기반 분류기"""
    
    def __init__(self, model_name='monologg/kobert', num_labels=6, dropout=0.3):
        super(KoBERTClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


def train_baseline1_tfidf_lr(train_texts, train_labels, val_texts, val_labels, 
                             test_texts, test_labels):
    """
    Baseline 1: TF-IDF + Logistic Regression
    """
    print("\n=== Training Baseline 1: TF-IDF + Logistic Regression ===")
    
    # 전처리
    train_texts_processed = preprocess_batch(train_texts)
    val_texts_processed = preprocess_batch(val_texts)
    test_texts_processed = preprocess_batch(test_texts)
    
    # TF-IDF 특징 추출
    extractor = TfidfFeatureExtractor(ngram_range=(1, 2), max_features=5000)
    X_train = extractor.fit_transform(train_texts_processed)
    X_val = extractor.transform(val_texts_processed)
    X_test = extractor.transform(test_texts_processed)
    
    # 라벨 인코딩
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_val = le.transform(val_labels)
    y_test = le.transform(test_labels)
    
    # 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # 모델 학습
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight_dict,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 예측
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    return {
        'model': model,
        'label_encoder': le,
        'extractor': extractor,
        'train_pred': train_pred,
        'val_pred': val_pred,
        'test_pred': test_pred,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


def train_baseline2_kobert_svm(train_texts, train_labels, val_texts, val_labels,
                               test_texts, test_labels, device='cpu'):
    """
    Baseline 2: KoBERT 임베딩 + SVM
    """
    print("\n=== Training Baseline 2: KoBERT Embedding + SVM ===")
    
    # 전처리
    train_texts_processed = preprocess_batch(train_texts)
    val_texts_processed = preprocess_batch(val_texts)
    test_texts_processed = preprocess_batch(test_texts)
    
    # KoBERT 임베딩 추출
    extractor = KoBERTFeatureExtractor(device=device)
    print("Extracting KoBERT embeddings for train set...")
    X_train = extractor.extract_cls_embedding(train_texts_processed)
    print("Extracting KoBERT embeddings for val set...")
    X_val = extractor.extract_cls_embedding(val_texts_processed)
    print("Extracting KoBERT embeddings for test set...")
    X_test = extractor.extract_cls_embedding(test_texts_processed)
    
    # 라벨 인코딩
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_val = le.transform(val_labels)
    y_test = le.transform(test_labels)
    
    # 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # SVM 모델 학습
    model = SVC(
        kernel='linear',
        class_weight=class_weight_dict,
        random_state=42
    )
    print("Training SVM...")
    model.fit(X_train, y_train)
    
    # 예측
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    return {
        'model': model,
        'label_encoder': le,
        'extractor': extractor,
        'train_pred': train_pred,
        'val_pred': val_pred,
        'test_pred': test_pred,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


def train_baseline3_kobert_finetune(train_texts, train_labels, val_texts, val_labels,
                                   test_texts, test_labels, device='cpu', 
                                   batch_size=16, epochs=3, lr=2e-5):
    """
    Baseline 3: KoBERT Fine-tuning
    """
    print("\n=== Training Baseline 3: KoBERT Fine-tuning ===")
    
    # 전처리
    train_texts_processed = preprocess_batch(train_texts)
    val_texts_processed = preprocess_batch(val_texts)
    test_texts_processed = preprocess_batch(test_texts)
    
    # 라벨 인코딩
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_val = le.transform(val_labels)
    y_test = le.transform(test_labels)
    
    num_labels = len(le.classes_)
    
    # 토크나이저 및 데이터셋
    tokenizer = AutoTokenizer.from_pretrained('monologg/kobert', trust_remote_code=True)
    train_dataset = CommentDataset(train_texts_processed, y_train, tokenizer)
    val_dataset = CommentDataset(val_texts_processed, y_val, tokenizer)
    test_dataset = CommentDataset(test_texts_processed, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델 초기화
    model = KoBERTClassifier(num_labels=num_labels)
    model.to(device)
    
    # 옵티마이저 및 스케줄러
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 학습
    best_val_loss = float('inf')
    train_pred_list = []
    val_pred_list = []
    test_pred_list = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 학습
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # 검증
        model.eval()
        val_loss = 0
        val_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
        
        val_pred_list = val_preds
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'results/model/kobert_finetune_best.pt')
    
    # 최종 예측 (학습 및 테스트 세트)
    model.load_state_dict(torch.load('results/model/kobert_finetune_best.pt'))
    model.eval()
    
    # 학습 세트 예측
    train_preds = []
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
    train_pred_list = train_preds
    
    # 테스트 세트 예측
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
    test_pred_list = test_preds
    
    return {
        'model': model,
        'label_encoder': le,
        'tokenizer': tokenizer,
        'train_pred': np.array(train_pred_list),
        'val_pred': np.array(val_pred_list),
        'test_pred': np.array(test_pred_list),
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       choices=['baseline1', 'baseline2', 'baseline3', 'all'],
                       help='Model to train')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for fine-tuning')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate for fine-tuning')
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # 데이터 로드
    df = load_labeled_comments()
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.2)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 결과 저장 디렉토리 생성
    ensure_dir('results/model')
    
    # 모델 학습
    results = {}
    
    if args.model in ['baseline1', 'all']:
        result1 = train_baseline1_tfidf_lr(
            train_df['comment_text'].tolist(),
            train_df['label'].tolist(),
            val_df['comment_text'].tolist(),
            val_df['label'].tolist(),
            test_df['comment_text'].tolist(),
            test_df['label'].tolist()
        )
        results['baseline1'] = result1
    
    if args.model in ['baseline2', 'all']:
        result2 = train_baseline2_kobert_svm(
            train_df['comment_text'].tolist(),
            train_df['label'].tolist(),
            val_df['comment_text'].tolist(),
            val_df['label'].tolist(),
            test_df['comment_text'].tolist(),
            test_df['label'].tolist(),
            device=device
        )
        results['baseline2'] = result2
    
    if args.model in ['baseline3', 'all']:
        result3 = train_baseline3_kobert_finetune(
            train_df['comment_text'].tolist(),
            train_df['label'].tolist(),
            val_df['comment_text'].tolist(),
            val_df['label'].tolist(),
            test_df['comment_text'].tolist(),
            test_df['label'].tolist(),
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr
        )
        results['baseline3'] = result3
    
    # 모델 저장
    for model_name, result in results.items():
        model_path = f'results/model/{model_name}.pkl'
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved {model_name} to {model_path}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

