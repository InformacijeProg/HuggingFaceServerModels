@echo off
rem Opens the hfserve chat window (starts the API server if needed) without a console window.
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
if exist "%ROOT%.vemv\Scripts\pythonw.exe" (
  start "" "%ROOT%.vemv\Scripts\pythonw.exe" -m hfserve gui
) else if exist "%ROOT%.venv\Scripts\pythonw.exe" (
  start "" "%ROOT%.venv\Scripts\pythonw.exe" -m hfserve gui
) else (
  start "" pythonw -m hfserve gui
)
endlocal
