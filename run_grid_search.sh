#!/bin/bash

# ==========================================
# Grid Search Configuration
# ==========================================
# MODELS=("EASE" "RLAE" "DLAE" "LAE" "DAN_EASE" "DAN_RLAE" "DAN_DLAE" "DAN_LAE" "ASPIRE_RLAE" "ASPIRE_EASE" "ASPIRE_DLAE" "ASPIRE_LAE")
MODELS=("ASPIRE_LAE" "ASPIRE_DLAE" "ASPIRE_RLAE" "ASPIRE_EASE")
# Datasets to run
DATASETS=("steam")

# Common Settings
GPU_ID=0
MODE="strong"

echo "=========================================="
echo "Starting Comprehensive Grid Search"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="

for M in "${MODELS[@]}"; do
    for D in "${DATASETS[@]}"; do
        echo ""
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] --- Running Model: $M / Dataset: $D"
        
        case $M in
            "EASE"|"LAE")
                # EASE/LAE: 10 ~ 10000
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_p_grid 10.0 10000.0 10 log
                ;;
            "RLAE")
                # RLAE: 10 ~ 1000
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_p_grid 10.0 1000.0 7 log \
                    --xi_grid 0.0 1.0 11 linear
                ;;
            "DLAE")
                # DLAE: 10 ~ 1000
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_p_grid 10.0 1000.0 7 log \
                    --dropout_grid 0.1 0.9 9 linear
                ;;
            "DAN_EASE"|"DAN_LAE")
                # DAN: 0.01 ~ 100
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_p_grid 0.01 100.0 9 log \
                    --alpha_grid 0.0 0.5 6 linear \
                    --beta_grid 0.0 1.0 11 linear
                ;;
            "DAN_RLAE")
                # DAN: 0.01 ~ 100
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_p_grid 0.01 100.0 7 log \
                    --alpha_grid 0.0 0.5 3 linear \
                    --beta_grid 0.0 1.0 6 linear \
                    --xi_grid 0.0 1.0 6 linear
                ;;
            "DAN_DLAE")
                # DAN: 0.01 ~ 100
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_p_grid 0.01 100.0 7 log \
                    --alpha_grid 0.0 0.5 3 linear \
                    --beta_grid 0.0 1.0 6 linear \
                    --dropout_grid 0.1 0.9 5 linear
                ;;
            "ASPIRE_EASE"|"ASPIRE_LAE")
                # ASPIRE: 0.01 ~ 100
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_lambda_grid 0.01 100.0 9 log \
                    --alpha_grid 0.0 1.0 11 linear
                ;;
            "ASPIRE_RLAE")
                # ASPIRE: 0.01 ~ 100
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_lambda_grid 0.01 100.0 7 log \
                    --alpha_grid 0.0 1.0 6 linear \
                    --xi_grid 0.0 1.0 6 linear
                ;;
            "ASPIRE_DLAE")
                # ASPIRE: 0.01 ~ 100
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_lambda_grid 0.01 100.0 7 log \
                    --alpha_grid 0.0 1.0 6 linear \
                    --dropout_grid 0.1 0.9 5 linear
                ;;
        esac
        
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] Completed $M on $D."
        else
            echo "[ERROR] Failed $M on $D with exit code $?"
        fi
    done
done

echo ""
echo "=========================================="
echo "All Grid Searches Complete!"
echo "Results are in 'results/' directory."
echo "=========================================="
