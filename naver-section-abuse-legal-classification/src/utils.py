"""
유틸리티 함수 모듈
"""
import os
import json
from typing import Dict, Any
import pandas as pd


def ensure_dir(directory: str):
    """디렉토리가 없으면 생성"""
    os.makedirs(directory, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str):
    """데이터를 JSON 파일로 저장"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results_summary(results: Dict[str, Any], output_path: str = "results/summary.json"):
    """실험 결과를 JSON으로 저장"""
    ensure_dir(os.path.dirname(output_path))
    save_json(results, output_path)
    print(f"Results saved to {output_path}")

