#!/bin/bash

# ==============================================================================
# ASPIRE-EASE Gamma Sweep Experiment
# ------------------------------------------------------------------------------
# Objective: Observe the performance trend as Gamma (alpha) changes.
# Gamma: 0.0 to 1.0 (Step 0.1, Total 11 points)
# Lambda: 1.0 to 100.0 (Log scale, 5 points HPO)
# ==============================================================================

# Ensure we are in the root directory
cd "$(dirname "$0")/.."

MODELS=("ASPIRE_EASE")
DATASETS=("steam")
GPU_ID=0
MODE="strong"
SAVE_DIR="exp_result"

echo "=========================================="
echo "Starting ASPIRE-EASE Gamma Sweep"
echo "Gamma Range: 0.0 - 1.0 (11 steps)"
echo "Lambda HPO: 1.0 - 100.0 (5 points, Log scale)"
echo "Results will be saved in: $SAVE_DIR"
echo "=========================================="

for D in "${DATASETS[@]}"; do
    echo ""
    echo "Running sweep on Dataset: $D"
    
    uv run python grid_search.py \
        --model ASPIRE_EASE \
        --dataset "$D" \
        --gpu $GPU_ID \
        --mode "$MODE" \
        --save_dir "$SAVE_DIR" \
        --alpha_grid 0.0 1.0 11 linear \
        --reg_lambda_grid 1.0 100.0 5 log
done

echo ""
echo "=========================================="
echo "Sweep Complete! Results are in '$SAVE_DIR/'"
echo "=========================================="
