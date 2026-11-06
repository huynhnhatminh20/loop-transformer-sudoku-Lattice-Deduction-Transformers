@echo off
REM Launcher script for the LDT / HRM / TRM Sudoku Gradio Demo
setlocal
cd /d "%~dp0\.."
python gradio_demo\app.py %*
pause
