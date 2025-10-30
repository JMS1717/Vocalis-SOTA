@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR%"=="" set "SCRIPT_DIR=%CD%\"

pushd "%SCRIPT_DIR%" >nul

set "ENV_FILE=%SCRIPT_DIR%.env"
set "ENV_HELPER=python ""%SCRIPT_DIR%scripts\manage_env.py"""
call :ensure_hf_token

if not exist "env\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.bat before starting Vocalis.
    popd >nul
    exit /b 1
)

echo === Starting Vocalis ===

echo === Ensuring required speech models are available ===
call "env\Scripts\activate.bat"
python scripts\download_models.py
if errorlevel 1 (
    echo [WARN] Some models failed to download. Check your Hugging Face credentials or gated model access.
)
call deactivate >nul 2>&1

set "BACKEND_TITLE=Vocalis Backend"
set "FRONTEND_TITLE=Vocalis Frontend"

start "%BACKEND_TITLE%" cmd /k "cd /d \"%SCRIPT_DIR%\" && call env\Scripts\activate.bat && python -m backend.main"

rem Allow backend a moment to start before launching the frontend
ping 127.0.0.1 -n 3 >nul

start "%FRONTEND_TITLE%" cmd /k "cd /d \"%SCRIPT_DIR%frontend\" && npm run dev"

echo === Vocalis servers started ===
echo Frontend: http://localhost:5173 (or your Vite port)
echo Backend:  http://localhost:8000 (or your FastAPI port)

popd >nul
exit /b 0

:ensure_hf_token
set "CURRENT_TOKEN=%HUGGINGFACE_TOKEN%"

if not defined CURRENT_TOKEN (
    for /f "usebackq tokens=*" %%T in (`%ENV_HELPER% --path "%ENV_FILE%" --get HUGGINGFACE_TOKEN 2^>nul`) do (
        set "CURRENT_TOKEN=%%T"
    )
    if defined CURRENT_TOKEN set "HUGGINGFACE_TOKEN=%CURRENT_TOKEN%"
)

if defined CURRENT_TOKEN (
    echo Using Hugging Face token from configuration.
    goto :eof
)

echo.
set /p NEW_HF_TOKEN="Enter your Hugging Face token for gated models (leave blank to continue without): "
if "%NEW_HF_TOKEN%"=="" (
    echo [WARN] Hugging Face token not provided. Model downloads for gated repositories may fail.
    goto :eof
)

set "HUGGINGFACE_TOKEN=%NEW_HF_TOKEN%"
setx HUGGINGFACE_TOKEN "%NEW_HF_TOKEN%" >nul
%ENV_HELPER% --path "%ENV_FILE%" --set HUGGINGFACE_TOKEN "%NEW_HF_TOKEN%"
echo Hugging Face token stored for future runs.
goto :eof
