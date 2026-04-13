@echo off
setlocal enabledelayedexpansion

:: 배치 파일이 있는 위치(루트)로 강제 이동
cd /d "%~dp0"

:: ==========================================
:: Grid Search Configuration
:: ==========================================
:: Models to run: CLAE, EASE, RLAE, GFCF, EASE_DAN, IPS_LAE
set MODELS=CLAE EASE RLAE GFCF EASE_DAN IPS_LAE
:: Datasets to run
set DATASETS=ml-20m msd netflix yelp2018 gowalla amazon-book 

:: Common Settings
set GPU_ID=0
set MODE=strong

echo ==========================================
echo Starting Comprehensive Grid Search from: %cd%
echo Models: %MODELS%
echo Datasets: %DATASETS%
echo ==========================================

for %%M in (%MODELS%) do (
    for %%D in (%DATASETS%) do (
        echo.
        echo [!time!] --- Running Model: %%M / Dataset: %%D
        
        if "%%M"=="CLAE" (
            uv run python grid_search.py --model CLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.1 1000.0 5 log ^
                --alpha_grid 0.1 1.0 10 linear ^
                --beta_grid 0.5 0.5 1 linear
        )
        if "%%M"=="EASE" (
            uv run python grid_search.py --model EASE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 10.0 10000.0 10 log
        )
        if "%%M"=="RLAE" (
            uv run python grid_search.py --model RLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.1 1000.0 5 log ^
                --xi_grid 0.0 1.0 5 linear
        )
        if "%%M"=="GFCF" (
            uv run python grid_search.py --model GFCF --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --alpha_grid 0.0 1.0 11 linear
        )
        if "%%M"=="EASE_DAN" (
            uv run python grid_search.py --model EASE_DAN --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 10.0 5000.0 5 log ^
                --alpha_grid 0.0 1.0 3 linear ^
                --beta_grid 0.0 1.0 3 linear
        )
        if "%%M"=="IPS_LAE" (
            :: IPS_LAE: logsigmoid mode
            uv run python grid_search.py --model IPS_LAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.1 1000.0 5 log ^
                --wbeta_grid 0.1 1.0 5 linear ^
                --wtype logsigmoid
                
            :: IPS_LAE: powerlaw mode
            uv run python grid_search.py --model IPS_LAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.1 1000.0 5 log ^
                --wbeta_grid 0.1 1.0 5 linear ^
                --wtype powerlaw
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
