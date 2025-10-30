@echo off
echo === Setting up Vocalis project ===

echo === Setting up frontend ===
cd frontend
call npm install
cd ..

set "SCRIPT_DIR=%~dp0"
set "ENV_HELPER=python ""%SCRIPT_DIR%scripts\manage_env.py"""
set "ENV_FILE=%SCRIPT_DIR%.env"

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

call :configure_hf_token

echo === Setup complete! ===
echo Run 'run.bat' to start the application
exit /b 0

:configure_hf_token
set "CURRENT_TOKEN=%HUGGINGFACE_TOKEN%"

if not defined CURRENT_TOKEN (
    for /f "usebackq tokens=*" %%T in (`%ENV_HELPER% --path "%ENV_FILE%" --get HUGGINGFACE_TOKEN 2^>nul`) do (
        set "CURRENT_TOKEN=%%T"
    )
)

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
%ENV_HELPER% --path "%ENV_FILE%" --set HUGGINGFACE_TOKEN "%NEW_HF_TOKEN%"
echo Hugging Face token stored for future runs.
goto :eof
