"""
모델 평가 스크립트 (성능 비교, Confusion Matrix)
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.labeling_rules import LABELS, get_label_name

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
plt.rcParams['axes.unicode_minus'] = False


def calculate_metrics(y_true, y_pred, label_encoder):
    """
    성능 지표 계산
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        label_encoder: 라벨 인코더
        
    Returns:
        성능 지표 딕셔너리
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # 클래스별 F1 점수
    f1_per_class = f1_score(y_true, y_pred, average=None)
    class_names = label_encoder.classes_
    f1_dict = {class_names[i]: f1_per_class[i] for i in range(len(class_names))}
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'f1_per_class': f1_dict
    }


def plot_confusion_matrix(y_true, y_pred, label_encoder, model_name: str,
                         output_path: str = None):
    """
    Confusion Matrix 시각화
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        label_encoder: 라벨 인코더
        model_name: 모델 이름
        output_path: 저장 경로
    """
    cm = confusion_matrix(y_true, y_pred)
    class_names = [get_label_name(label) for label in label_encoder.classes_]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('실제 라벨', fontsize=12)
    plt.xlabel('예측 라벨', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model(model_result, model_name: str, split: str = 'test',
                  output_dir: str = 'results'):
    """
    모델 평가 수행
    
    Args:
        model_result: 모델 학습 결과 딕셔너리
        model_name: 모델 이름
        split: 평가할 데이터셋 ('train', 'val', 'test')
        output_dir: 출력 디렉토리
        
    Returns:
        평가 결과 딕셔너리
    """
    y_true = model_result[f'y_{split}']
    y_pred = model_result[f'{split}_pred']
    label_encoder = model_result['label_encoder']
    
    # 성능 지표 계산
    metrics = calculate_metrics(y_true, y_pred, label_encoder)
    
    # Classification Report
    class_names = [get_label_name(label) for label in label_encoder.classes_]
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion Matrix 시각화
    cm_path = os.path.join(output_dir, 'charts', f'cm_{model_name}_{split}.png')
    plot_confusion_matrix(y_true, y_pred, label_encoder, 
                         f'{model_name} ({split})', cm_path)
    
    return {
        'metrics': metrics,
        'classification_report': report
    }


def compare_models(model_results: dict, split: str = 'test',
                  output_dir: str = 'results'):
    """
    여러 모델의 성능 비교
    
    Args:
        model_results: 모델별 결과 딕셔너리
        split: 비교할 데이터셋 ('train', 'val', 'test')
        output_dir: 출력 디렉토리
        
    Returns:
        비교 결과 DataFrame
    """
    comparison_data = []
    
    for model_name, result in model_results.items():
        y_true = result[f'y_{split}']
        y_pred = result[f'{split}_pred']
        label_encoder = result['label_encoder']
        
        metrics = calculate_metrics(y_true, y_pred, label_encoder)
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Macro F1': metrics['macro_f1'],
            'Weighted F1': metrics['weighted_f1']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 비교 테이블 저장
    comparison_path = os.path.join(output_dir, f'model_comparison_{split}.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\n=== Model Comparison ({split}) ===")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved to {comparison_path}")
    
    # 비교 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.25
    
    ax.bar(x - width, comparison_df['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x, comparison_df['Macro F1'], width, label='Macro F1', alpha=0.8)
    ax.bar(x + width, comparison_df['Weighted F1'], width, label='Weighted F1', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Performance Comparison ({split})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'])
    ax.legend()
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'charts', f'model_comparison_{split}.png')
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison chart to {chart_path}")
    plt.close()
    
    return comparison_df


def generate_evaluation_report(model_results: dict, output_dir: str = 'results'):
    """
    전체 평가 리포트 생성
    
    Args:
        model_results: 모델별 결과 딕셔너리
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'charts'), exist_ok=True)
    
    all_results = {}
    
    # 각 모델별 평가
    for model_name, result in model_results.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        model_results_dict = {}
        
        for split in ['train', 'val', 'test']:
            eval_result = evaluate_model(result, model_name, split, output_dir)
            model_results_dict[split] = eval_result
            
            print(f"\n--- {split.upper()} Set ---")
            print(f"Accuracy: {eval_result['metrics']['accuracy']:.4f}")
            print(f"Macro F1: {eval_result['metrics']['macro_f1']:.4f}")
            print(f"Weighted F1: {eval_result['metrics']['weighted_f1']:.4f}")
        
        all_results[model_name] = model_results_dict
    
    # 모델 비교
    print(f"\n{'='*50}")
    print("Model Comparison")
    print(f"{'='*50}")
    
    for split in ['train', 'val', 'test']:
        compare_models(model_results, split, output_dir)
    
    # 전체 리포트 저장
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 모델 평가 리포트 ===\n\n")
        
        for model_name, results_dict in all_results.items():
            f.write(f"\n{model_name}\n")
            f.write("-" * 50 + "\n")
            
            for split, eval_result in results_dict.items():
                f.write(f"\n{split.upper()} Set:\n")
                f.write(f"  Accuracy: {eval_result['metrics']['accuracy']:.4f}\n")
                f.write(f"  Macro F1: {eval_result['metrics']['macro_f1']:.4f}\n")
                f.write(f"  Weighted F1: {eval_result['metrics']['weighted_f1']:.4f}\n")
    
    print(f"\nSaved evaluation report to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='results/model',
                       help='Directory containing saved models')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['baseline1', 'baseline2', 'baseline3'],
                       help='Models to evaluate')
    args = parser.parse_args()
    
    # 모델 로드
    model_results = {}
    for model_name in args.models:
        model_path = os.path.join(args.model_dir, f'{model_name}.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_results[model_name] = pickle.load(f)
            print(f"Loaded {model_name} from {model_path}")
        else:
            print(f"Warning: {model_path} not found")
    
    if not model_results:
        print("No models found to evaluate")
        return
    
    # 평가 리포트 생성
    generate_evaluation_report(model_results)


if __name__ == "__main__":
    main()

