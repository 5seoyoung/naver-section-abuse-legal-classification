#!/bin/bash

# 모델 학습 실행 스크립트

MODEL=${1:-all}  # 기본값: all
DEVICE=${2:-cpu}  # 기본값: cpu

echo "=== 모델 학습 시작 ==="
echo "Model: $MODEL"
echo "Device: $DEVICE"

python -m src.train_model --model $MODEL --device $DEVICE

