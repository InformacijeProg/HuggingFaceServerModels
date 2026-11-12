@echo off
rem Opens the hftrain window (training with progress bar and console, chat) without a console window.
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
if exist "%ROOT%.vemv\Scripts\pythonw.exe" (
  start "" "%ROOT%.vemv\Scripts\pythonw.exe" -m hftrain gui
) else if exist "%ROOT%.venv\Scripts\pythonw.exe" (
  start "" "%ROOT%.venv\Scripts\pythonw.exe" -m hftrain gui
) else (
  start "" pythonw -m hftrain gui
)
endlocal
