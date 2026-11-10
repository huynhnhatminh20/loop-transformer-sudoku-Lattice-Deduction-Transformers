@echo off
REM Launcher for the LDT / HRM / TRM Sudoku Gradio Demo.
REM Tries native Windows Python first, then falls back to WSL when Windows
REM Application Control (Smart App Control) blocks torch/ujson DLLs.
setlocal
cd /d "%~dp0\.."

where python >nul 2>nul
if errorlevel 1 (
  echo [WARN] Python not found in PATH. Falling back to WSL...
  call "%~dp0run_demo_wsl.bat" %*
  exit /b %errorlevel%
)

echo [CHECK] Checking native Python packages...
python -c "import torch, gradio, numpy" >nul 2>&1
if errorlevel 1 (
  echo [WARN] Native Python cannot import torch/gradio/numpy.
  echo        Missing packages, or Smart App Control is blocking unsigned DLLs
  echo        ^(WinError 4551^). Falling back to WSL...
  call "%~dp0run_demo_wsl.bat" %*
  exit /b %errorlevel%
)

echo [RUN] Starting Gradio demo with native Python...
python gradio_demo\app.py %*
pause
