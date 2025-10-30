@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR%"=="" set "SCRIPT_DIR=%CD%\"

pushd "%SCRIPT_DIR%" >nul

set "ENV_FILE=%SCRIPT_DIR%.env"

if not defined HUGGINGFACE_TOKEN if exist "%ENV_FILE%" (
    for /f "usebackq tokens=1* delims==" %%A in (`findstr /B "HUGGINGFACE_TOKEN=" "%ENV_FILE%" 2^>nul`) do (
        if /I "%%A"=="HUGGINGFACE_TOKEN" (
            set "HUGGINGFACE_TOKEN=%%B"
        )
    )
)

if not defined HUGGINGFACE_TOKEN (
    echo.
    set /p HUGGINGFACE_TOKEN="Enter your Hugging Face token for gated models (leave blank to continue without): "
    if not "%HUGGINGFACE_TOKEN%"=="" (
        setx HUGGINGFACE_TOKEN "%HUGGINGFACE_TOKEN%" >nul
        if exist "%ENV_FILE%" (
            set "HF_TOKEN_PRESENT="
            for /f "usebackq tokens=1* delims==" %%A in (`findstr /B "HUGGINGFACE_TOKEN=" "%ENV_FILE%" 2^>nul`) do (
                if /I "%%A"=="HUGGINGFACE_TOKEN" set "HF_TOKEN_PRESENT=1"
            )
            if not defined HF_TOKEN_PRESENT (
                echo HUGGINGFACE_TOKEN=%HUGGINGFACE_TOKEN%>>"%ENV_FILE%"
            )
        ) else (
            echo HUGGINGFACE_TOKEN=%HUGGINGFACE_TOKEN%>"%ENV_FILE%"
        )
        echo Hugging Face token stored for future runs.
    ) else (
        echo [WARN] Hugging Face token not provided. Model downloads for gated repositories may fail.
    )
)

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

start "%BACKEND_TITLE%" cmd /k "cd /d \"%SCRIPT_DIR%\" && call env\Scripts\activate && python -m backend.main"

rem Allow backend a moment to start before launching the frontend
ping 127.0.0.1 -n 3 >nul

start "%FRONTEND_TITLE%" cmd /k "cd /d \"%SCRIPT_DIR%frontend\" && npm run dev"

echo === Vocalis servers started ===
echo Frontend: http://localhost:5173 (or your Vite port)
echo Backend:  http://localhost:8000 (or your FastAPI port)

popd >nul
exit /b 0
