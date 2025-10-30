@echo off
echo === Setting up Vocalis project ===

echo === Setting up frontend ===
cd frontend
call npm install
cd ..

echo === Setting up backend environment ===
python -m venv env
call .\env\Scripts\activate

echo.
echo Would you like to install PyTorch with CUDA support?
echo 1. Yes - Install with CUDA support (recommended for NVIDIA GPUs)
echo 2. No - Use CPU only
choice /c 12 /n /m "Enter your choice (1 or 2): "

if errorlevel 2 (
    echo === Installing with CPU support only ===
    python -m pip install -r backend\requirements.txt
) else (
    echo === Installing with CUDA support ===
    python -m pip install -r backend\requirements.txt
    echo === Installing PyTorch with CUDA support ===
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
)

set "ENV_FILE=%~dp0.env"
call :configure_hf_token

echo === Setup complete! ===
echo Run 'run.bat' to start the application
exit /b 0

:configure_hf_token
set "ENV_FILE_TOKEN="
set "CURRENT_TOKEN=%HUGGINGFACE_TOKEN%"
if exist "%ENV_FILE%" (
    for /f "usebackq tokens=1* delims==" %%A in (`findstr /B "HUGGINGFACE_TOKEN=" "%ENV_FILE%" 2^>nul`) do (
        if /I "%%A"=="HUGGINGFACE_TOKEN" set "ENV_FILE_TOKEN=%%B"
    )
)
if not defined CURRENT_TOKEN if defined ENV_FILE_TOKEN set "CURRENT_TOKEN=%ENV_FILE_TOKEN%"
if defined CURRENT_TOKEN (
    echo Hugging Face token already configured.
    choice /c YN /n /m "Do you want to update it? (Y/N): "
    if errorlevel 2 (
        echo Keeping existing Hugging Face token configuration.
        goto :eof
    )
)
set /p NEW_HF_TOKEN="Enter your Hugging Face token (leave blank to skip): "
if "%NEW_HF_TOKEN%"=="" (
    if defined CURRENT_TOKEN (
        echo Keeping existing Hugging Face token configuration.
    ) else (
        echo Skipping Hugging Face token configuration.
    )
    goto :eof
)
set "HUGGINGFACE_TOKEN=%NEW_HF_TOKEN%"
setx HUGGINGFACE_TOKEN "%NEW_HF_TOKEN%" >nul
call :write_env_token "%NEW_HF_TOKEN%"
echo Hugging Face token stored for future runs.
goto :eof

:write_env_token
set "NEW_TOKEN=%~1"
if not exist "%ENV_FILE%" (
    type nul > "%ENV_FILE%"
) else (
    powershell -NoLogo -NoProfile -Command ^
        "$path = [System.IO.Path]::GetFullPath('%ENV_FILE%');" ^
        "if (Test-Path $path) {" ^
        "  $lines = Get-Content -Path $path;" ^
        "  $lines | Where-Object {$_ -notlike 'HUGGINGFACE_TOKEN=*'} | Set-Content -Path $path;" ^
        "}" >nul 2>&1
)
>>"%ENV_FILE%" echo HUGGINGFACE_TOKEN=%NEW_TOKEN%
goto :eof
