@echo off
setlocal enabledelayedexpansion

:: 배치 파일이 있는 위치(루트)로 강제 이동
cd /d "%~dp0"

:: ==========================================
:: Grid Search Configuration (Balanced Budget ~100 trials/model)
:: ==========================================
:: set MODELS=EASE RLAE DLAE LAE DAN_EASE DAN_RLAE DAN_DLAE DAN_LAE ASPIRE_RLAE ASPIRE_EASE ASPIRE_DLAE ASPIRE_LAE DAspire_EASE DAspire_LAE DAspire_RLAE DAspire_DLAE IPS_LAE IPS_EASE IPS_RLAE IPS_DLAE
set MODELS=DAspire_EASE DAspire_LAE DAspire_RLAE DAspire_DLAE
:: Datasets to run
set DATASETS=steam

:: Common Settings
set GPU_ID=0
set MODE=strong

echo ==========================================
echo Starting Balanced Grid Search (Budget Control)
echo Models: %MODELS%
echo Datasets: %DATASETS%
echo ==========================================

for %%M in (%MODELS%) do (
    for %%D in (%DATASETS%) do (
        echo.
        echo [!time!] --- Running Model: %%M / Dataset: %%D
        
        if "%%M"=="EASE" (
            :: 1 param: 20 trials
            uv run python grid_search.py --model EASE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 10.0 10000.0 20 log
        )
        if "%%M"=="LAE" (
            :: 1 param: 20 trials
            uv run python grid_search.py --model LAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 10.0 10000.0 20 log
        )
        if "%%M"=="RLAE" (
            :: 2 params: 10 x 10 = 100 trials
            uv run python grid_search.py --model RLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 10.0 1000.0 10 log ^
                --xi_grid 0.0 1.0 10 linear
        )
        if "%%M"=="DLAE" (
            :: 1 param: 20 trials
            uv run python grid_search.py --model DLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --dropout_grid 0.1 0.9 10 linear
        )
        if "%%M"=="DAN_EASE" (
            :: 3 params: 5 x 4 x 5 = 100 trials
            uv run python grid_search.py --model DAN_EASE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 0.1 100.0 5 log ^
                --alpha_grid 0.0 0.5 4 linear ^
                --beta_grid 0.0 1.0 5 linear
        )
        if "%%M"=="DAN_LAE" (
            :: 3 params: 5 x 4 x 5 = 100 trials
            uv run python grid_search.py --model DAN_LAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 0.1 100.0 5 log ^
                --alpha_grid 0.0 0.5 4 linear ^
                --beta_grid 0.0 1.0 5 linear
        )
        if "%%M"=="DAN_RLAE" (
            :: 4 params: 4 x 3 x 3 x 3 = 108 trials
            uv run python grid_search.py --model DAN_RLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 0.1 100.0 4 log ^
                --alpha_grid 0.0 0.5 3 linear ^
                --beta_grid 0.0 1.0 3 linear ^
                --xi_grid 0.0 1.0 3 linear
        )
        if "%%M"=="DAN_DLAE" (
            :: 3 params: 4 x 5 x 5 = 100 trials
            uv run python grid_search.py --model DAN_DLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --alpha_grid 0.0 0.5 4 linear ^
                --beta_grid 0.0 1.0 5 linear ^
                --dropout_grid 0.1 0.9 5 linear
        )
        if "%%M"=="ASPIRE_EASE" (
            :: 2 params: 10 x 10 = 100 trials
            uv run python grid_search.py --model ASPIRE_EASE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.1 100.0 10 log ^
                --alpha_grid 0.1 1.0 10 linear
        )
        if "%%M"=="ASPIRE_LAE" (
            :: 2 params: 10 x 10 = 100 trials
            uv run python grid_search.py --model ASPIRE_LAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.1 100.0 10 log ^
                --alpha_grid 0.1 1.0 10 linear
        )
        if "%%M"=="ASPIRE_RLAE" (
            :: 3 params: 5 x 4 x 5 = 100 trials
            uv run python grid_search.py --model ASPIRE_RLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.1 100.0 5 log ^
                --alpha_grid 0.0 1.0 4 linear ^
                --xi_grid 0.0 1.0 5 linear
        )
        if "%%M"=="ASPIRE_DLAE" (
            :: 2 params: 10 x 9 = 90 trials
            uv run python grid_search.py --model ASPIRE_DLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --alpha_grid 0.1 1.0 10 linear ^
                --dropout_grid 0.1 0.9 9 linear
        )
        if "%%M"=="DAspire_EASE" (
            :: 3 params: 5 x 4 x 5 = 100 trials
            uv run python grid_search.py --model DAspire_EASE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.1 100.0 5 log ^
                --alpha_grid 0.0 0.5 4 linear ^
                --beta_grid 0.0 1.0 5 linear
        )
        if "%%M"=="DAspire_LAE" (
            :: 3 params: 5 x 4 x 5 = 100 trials
            uv run python grid_search.py --model DAspire_LAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.1 100.0 5 log ^
                --alpha_grid 0.0 0.5 4 linear ^
                --beta_grid 0.0 1.0 5 linear
        )
        if "%%M"=="DAspire_RLAE" (
            :: 4 params: 4 x 3 x 3 x 3 = 108 trials
            uv run python grid_search.py --model DAspire_RLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.1 100.0 4 log ^
                --alpha_grid 0.0 0.5 3 linear ^
                --beta_grid 0.0 1.0 3 linear ^
                --xi_grid 0.0 1.0 3 linear
        )
        if "%%M"=="DAspire_DLAE" (
            :: 3 params: 4 x 5 x 5 = 100 trials
            uv run python grid_search.py --model DAspire_DLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --alpha_grid 0.0 0.5 4 linear ^
                --beta_grid 0.0 1.0 5 linear ^
                --dropout_grid 0.1 0.9 5 linear
        )
        if "%%M"=="IPS_LAE" (
            :: 2 params: 10 x 10 = 100 trials
            uv run python grid_search.py --model IPS_LAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 10.0 10000.0 10 log ^
                --wbeta_grid 0.0 1.0 10 linear
        )
        if "%%M"=="IPS_EASE" (
            :: 2 params: 10 x 10 = 100 trials
            uv run python grid_search.py --model IPS_EASE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 10.0 10000.0 10 log ^
                --wbeta_grid 0.0 1.0 10 linear
        )
        if "%%M"=="IPS_RLAE" (
            :: 3 params: 5 x 5 x 4 = 100 trials
            uv run python grid_search.py --model IPS_RLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 10.0 1000.0 5 log ^
                --wbeta_grid 0.0 1.0 5 linear ^
                --xi_grid 0.0 0.9 4 linear
        )
        if "%%M"=="IPS_DLAE" (
            :: 2 params: 10 x 10 = 100 trials
            uv run python grid_search.py --model IPS_DLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --dropout_grid 0.1 0.9 10 linear ^
                --wbeta_grid 0.0 1.0 10 linear
        )
        
        if !errorlevel! equ 0 (
            echo [SUCCESS] Completed %%M on %%D.
        ) else (
            echo [ERROR] Failed %%M on %%D with exit code !errorlevel!
        )
    )
)

echo.
echo ==========================================
echo All Grid Searches Complete!
echo Results are in 'results/' directory.
echo ==========================================
pause
