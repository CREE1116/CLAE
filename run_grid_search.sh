#!/bin/bash

# ==========================================
# Grid Search Configuration (Balanced Budget ~100 trials/model)
# ==========================================
# MODELS=("EASE" "RLAE" "DLAE" "LAE" "DAN_EASE" "DAN_RLAE" "DAN_DLAE" "DAN_LAE" "ASPIRE_RLAE" "ASPIRE_EASE" "ASPIRE_DLAE" "ASPIRE_LAE" "IPS_LAE" "IPS_EASE" "IPS_RLAE" "IPS_DLAE")
MODELS=("IPS_LAE" "IPS_EASE" "IPS_DLAE")
# Datasets to run
DATASETS=("steam")

# Common Settings
GPU_ID=0
MODE="strong"

echo "=========================================="
echo "Starting Balanced Grid Search (Budget Control)"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="

for M in "${MODELS[@]}"; do
    for D in "${DATASETS[@]}"; do
        echo ""
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] --- Running Model: $M / Dataset: $D"
        
        case $M in
            "EASE"|"LAE")
                # 1 param: 20 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_p_grid 10.0 10000.0 20 log
                ;;
            "RLAE")
                # 2 params: 10 x 10 = 100 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_p_grid 10.0 1000.0 10 log \
                    --xi_grid 0.0 1.0 10 linear
                ;;
            "DLAE")
                # 1 param: 20 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --dropout_grid 0.1 0.9 10 linear
                ;;
            "DAN_EASE"|"DAN_LAE")
                # 3 params: 5 x 4 x 5 = 100 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_p_grid 0.01 100.0 5 log \
                    --alpha_grid 0.0 0.5 4 linear \
                    --beta_grid 0.0 1.0 5 linear
                ;;
            "DAN_RLAE")
                # 4 params: 4 x 3 x 3 x 3 = 108 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_p_grid 0.01 100.0 4 log \
                    --alpha_grid 0.0 0.5 3 linear \
                    --beta_grid 0.0 1.0 3 linear \
                    --xi_grid 0.0 1.0 3 linear
                ;;
            "DAN_DLAE")
                # 3 params: 4 x 5 x 5 = 100 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --alpha_grid 0.0 0.5 4 linear \
                    --beta_grid 0.0 1.0 5 linear \
                    --dropout_grid 0.1 0.9 5 linear
                ;;
            "ASPIRE_EASE"|"ASPIRE_LAE")
                # 2 params: 10 x 10 = 100 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_lambda_grid 0.1 100.0 10 log \
                    --alpha_grid 0.1 1.0 10 linear
                ;;
            "ASPIRE_RLAE")
                # 3 params: 5 x 4 x 5 = 100 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_lambda_grid 0.01 100.0 5 log \
                    --alpha_grid 0.0 1.0 4 linear \
                    --xi_grid 0.0 1.0 5 linear
                ;;
            "ASPIRE_DLAE")
                # 2 params: 10 x 9 = 90 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --alpha_grid 0.1 1.0 10 linear \
                    --dropout_grid 0.1 0.9 9 linear
                ;;
            "IPS_LAE"|"IPS_EASE")
                # 2 params: 10 x 10 = 100 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_lambda_grid 10.0 10000.0 10 log \
                    --wbeta_grid 0.0 1.0 10 linear
                ;;
            "IPS_RLAE")
                # 3 params: 5 x 5 x 4 = 100 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --reg_lambda_grid 10.0 1000.0 5 log \
                    --wbeta_grid 0.0 1.0 5 linear \
                    --xi_grid 0.0 0.9 4 linear
                ;;
            "IPS_DLAE")
                # 2 params: 10 x 10 = 100 trials
                uv run python grid_search.py --model "$M" --dataset "$D" --gpu $GPU_ID --mode "$MODE" \
                    --dropout_grid 0.1 0.9 10 linear \
                    --wbeta_grid 0.0 1.0 10 linear
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
