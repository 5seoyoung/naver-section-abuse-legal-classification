#!/bin/bash

# 모델 평가 실행 스크립트

MODELS=${1:-"baseline1 baseline2 baseline3"}

echo "=== 모델 평가 시작 ==="
echo "Models: $MODELS"

python -m src.evaluate --models $MODELS

