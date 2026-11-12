@echo off
rem Launcher for Windows: run hftrain (fine-tuning) with the project's virtualenv.
setlocal
set "ROOT=%~dp0"
if exist "%ROOT%.vemv\Scripts\python.exe" (
  "%ROOT%.vemv\Scripts\python.exe" -m hftrain %*
) else if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" -m hftrain %*
) else (
  python -m hftrain %*
)
endlocal
