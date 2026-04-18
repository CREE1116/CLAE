@echo off
setlocal enabledelayedexpansion

:: 실험 루트 디렉토리로 이동
cd /d "%~dp0\.."

:: ==============================================================================
:: ASPIRE-EASE Gamma Sweep Experiment (Windows)
:: ==============================================================================

set MODELS=ASPIRE_EASE
set DATASETS=steam
set GPU_ID=0
set MODE=strong
set SAVE_DIR=exp_result

echo ==========================================
echo Starting ASPIRE-EASE Gamma Sweep
echo Gamma Range: 0.0 - 1.0 (11 steps)
echo Lambda HPO: 1.0 - 100.0 (5 points, Log scale)
echo Results will be saved in: %SAVE_DIR%
echo ==========================================

for %%D in (%DATASETS%) do (
    echo.
    echo Running sweep on Dataset: %%D
    
    uv run python grid_search.py ^
        --model ASPIRE_EASE ^
        --dataset %%D ^
        --gpu %GPU_ID% ^
        --mode %MODE% ^
        --save_dir %SAVE_DIR% ^
        --alpha_grid 0.0 2.0 21 linear ^
        --reg_lambda_grid 1.0 100.0 5 log
)

echo.
echo ==========================================
echo Sweep Complete!
echo Results are in '%SAVE_DIR%/'
echo ==========================================
pause
