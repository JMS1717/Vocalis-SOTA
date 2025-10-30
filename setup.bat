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
set "HF_VAR_PRESENT="
if exist "%ENV_FILE%" (
    for /f "usebackq tokens=1* delims==" %%A in (`findstr /B "HUGGINGFACE_TOKEN=" "%ENV_FILE%" 2^>nul`) do (
        if /I "%%A"=="HUGGINGFACE_TOKEN" (
            set "HF_VAR_PRESENT=1"
        )
    )
)

if "%HUGGINGFACE_TOKEN%"=="" if not defined HF_VAR_PRESENT (
    echo.
    set /p HUGGINGFACE_TOKEN="Enter your Hugging Face token (leave blank to skip): "
    if not "%HUGGINGFACE_TOKEN%"=="" (
        if not exist "%ENV_FILE%" (
            type nul > "%ENV_FILE%"
        )
        setx HUGGINGFACE_TOKEN "%HUGGINGFACE_TOKEN%" >nul
        echo HUGGINGFACE_TOKEN=%HUGGINGFACE_TOKEN%>>"%ENV_FILE%"
        echo Hugging Face token stored in .env and user environment.
    ) else (
        echo Skipping Hugging Face token configuration.
    )
) else if defined HF_VAR_PRESENT (
    echo Hugging Face token already configured in .env.
) else (
    echo Using Hugging Face token from current environment.
)

echo === Setup complete! ===
echo Run 'run.bat' to start the application
