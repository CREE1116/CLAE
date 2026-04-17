@echo off
setlocal enabledelayedexpansion

:: 배치 파일이 있는 위치(루트)로 강제 이동
cd /d "%~dp0"

:: ==========================================
:: Grid Search Configuration
:: ==========================================
:: set MODELS=EASE RLAE DLAE LAE DAN_EASE DAN_RLAE DAN_DLAE DAN_LAE ASPIRE_RLAE ASPIRE_EASE ASPIRE_DLAE ASPIRE_LAE
set MODELS=ASPIRE_LAE ASPIRE_DLAE ASPIRE_RLAE ASPIRE_EASE
:: Datasets to run
set DATASETS=steam

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
        
        if "%%M"=="EASE" (
            :: EASE: 10 ~ 10000
            uv run python grid_search.py --model EASE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 10.0 10000.0 10 log
        )
        if "%%M"=="LAE" (
            :: LAE: 10 ~ 10000
            uv run python grid_search.py --model LAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 10.0 10000.0 10 log
        )
        if "%%M"=="RLAE" (
            :: RLAE: 10 ~ 1000
            uv run python grid_search.py --model RLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 10.0 1000.0 7 log ^
                --xi_grid 0.0 1.0 11 linear
        )
        if "%%M"=="DLAE" (
            :: DLAE: 10 ~ 1000
            uv run python grid_search.py --model DLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 10.0 1000.0 7 log ^
                --dropout_grid 0.1 0.9 9 linear
        )
        if "%%M"=="DAN_EASE" (
            :: DAN: 0.01 ~ 100
            uv run python grid_search.py --model DAN_EASE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 0.01 100.0 9 log ^
                --alpha_grid 0.0 0.5 6 linear ^
                --beta_grid 0.0 1.0 11 linear
        )
        if "%%M"=="DAN_LAE" (
            :: DAN: 0.01 ~ 100
            uv run python grid_search.py --model DAN_LAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 0.01 100.0 9 log ^
                --alpha_grid 0.0 0.5 6 linear ^
                --beta_grid 0.0 1.0 11 linear
        )
        if "%%M"=="DAN_RLAE" (
            :: DAN: 0.01 ~ 100
            uv run python grid_search.py --model DAN_RLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 0.01 100.0 7 log ^
                --alpha_grid 0.0 0.5 3 linear ^
                --beta_grid 0.0 1.0 6 linear ^
                --xi_grid 0.0 1.0 6 linear
        )
        if "%%M"=="DAN_DLAE" (
            :: DAN: 0.01 ~ 100
            uv run python grid_search.py --model DAN_DLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_p_grid 0.01 100.0 7 log ^
                --alpha_grid 0.0 0.5 3 linear ^
                --beta_grid 0.0 1.0 6 linear ^
                --dropout_grid 0.1 0.9 5 linear
        )
        if "%%M"=="ASPIRE_EASE" (
            :: ASPIRE: 0.01 ~ 100
            uv run python grid_search.py --model ASPIRE_EASE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.01 100.0 9 log ^
                --alpha_grid 0.0 1.0 11 linear
        )
        if "%%M"=="ASPIRE_LAE" (
            :: ASPIRE: 0.01 ~ 100
            uv run python grid_search.py --model ASPIRE_LAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.01 100.0 9 log ^
                --alpha_grid 0.0 1.0 11 linear
        )
        if "%%M"=="ASPIRE_RLAE" (
            :: ASPIRE: 0.01 ~ 100
            uv run python grid_search.py --model ASPIRE_RLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.01 100.0 7 log ^
                --alpha_grid 0.0 1.0 6 linear ^
                --xi_grid 0.0 1.0 6 linear
        )
        if "%%M"=="ASPIRE_DLAE" (
            :: ASPIRE: 0.01 ~ 100
            uv run python grid_search.py --model ASPIRE_DLAE --dataset %%D --gpu %GPU_ID% --mode %MODE% ^
                --reg_lambda_grid 0.01 100.0 7 log ^
                --alpha_grid 0.0 1.0 6 linear ^
                --dropout_grid 0.1 0.9 5 linear
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
