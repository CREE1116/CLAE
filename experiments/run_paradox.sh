#!/bin/bash

# Dataset list: abook, ml-20m, yelp2018, netflix, steam
DATASETS=("ml-20m")
GPU_ID=0

# 스크립트의 실제 위치를 파악하여 실행 경로를 고정합니다.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Running Paradox of Correction Field Validation"
echo "=========================================="

for D in "${DATASETS[@]}"; do
    echo ""
    echo ">>> Target Dataset: $D"
    uv run python "$SCRIPT_DIR/paradox_experiment.py" --dataset "$D" --gpu $GPU_ID
done

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "Check 'exp_result/' for CSVs and Plots."
echo "=========================================="
