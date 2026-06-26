@echo off
echo ========================================
echo Qwen30B Chat Build Script
echo ========================================

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Install dependencies
echo [1/4] Installing Python dependencies...
pip install PyQt6 pyinstaller -q

:: Create output directory
echo [2/4] Preparing output directory...
if not exist "dist" mkdir dist
if not exist "dist\model" mkdir dist\model

:: Package Python program
echo [3/4] Packaging GUI program...
pyinstaller --noconfirm --clean --name "Qwen30BChat" --windowed --onedir chat_gui.py

:: Copy necessary files
echo [4/4] Copying dependencies...
xcopy /Y /Q "llama-cli.exe" "dist\Qwen30BChat\"
xcopy /Y /Q "*.dll" "dist\Qwen30BChat\"

:: Create launch script
echo @echo off > dist\launch.bat
echo cd /d "%~dp0" >> dist\launch.bat
echo Qwen30BChat\Qwen30BChat.exe >> dist\launch.bat

:: Create readme
echo Qwen30B Chat Application > dist\readme.txt
echo ======================== >> dist\readme.txt
echo. >> dist\readme.txt
echo Usage: >> dist\readme.txt
echo 1. Run "launch.bat" to start the program >> dist\readme.txt
echo 2. Click "Download Model" button if model is missing >> dist\readme.txt
echo 3. Wait for model download (~18GB from China mirror) >> dist\readme.txt
echo 4. Start chatting after download completes >> dist\readme.txt
echo. >> dist\readme.txt
echo Requirements: >> dist\readme.txt
echo - Windows 10/11 64-bit >> dist\readme.txt
echo - NVIDIA GPU (recommended) or CPU >> dist\readme.txt
echo - 8GB+ RAM, 16GB recommended >> dist\readme.txt
echo. >> dist\readme.txt
echo Model: Qwen3-30B-A3B (Q4_K_M, ~18GB) >> dist\readme.txt
echo Engine: llama.cpp >> dist\readme.txt
echo GUI: PyQt6 >> dist\readme.txt

:: Create distribution package
echo.
echo Creating distribution package...
powershell -Command "Compress-Archive -Path 'dist\*' -DestinationPath 'dist\Qwen30BChat_Package.zip' -Force"

echo.
echo ========================================
echo Build Complete!
echo Output: dist\Qwen30BChat_Package.zip
echo ========================================
echo.
pause