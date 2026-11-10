@echo off
REM Fallback launcher: run the Gradio demo inside WSL (Linux).
REM Use this when Windows Application Control (Smart App Control) blocks
REM unsigned DLLs such as torch\lib\torch.dll or ujson.
setlocal
cd /d "%~dp0\.."
set "REPO_DIR=%~dp0.."

where wsl.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] WSL is not installed. Install it with:  wsl --install -d Ubuntu-22.04
  pause
  exit /b 1
)

echo [CHECK] Looking for torch, gradio, numpy in WSL...
wsl.exe --cd "%REPO_DIR%" -- python3 -c "import torch, gradio, numpy" >nul 2>&1
if errorlevel 1 (
  echo [SETUP] Installing torch, gradio, numpy inside WSL ^(one-time^)...
  wsl.exe -- python3 -m pip install --user --progress-bar off torch gradio numpy
  if errorlevel 1 (
    echo [ERROR] Package install failed. See README section 1.4.
    pause
    exit /b 1
  )
)

echo [RUN] Starting Gradio demo from WSL...
wsl.exe --cd "%REPO_DIR%" -- python3 gradio_demo/app.py %*
pause
