@echo off
rem Launcher for Windows: run hfserve with the project's virtualenv.
setlocal
set "ROOT=%~dp0"
if exist "%ROOT%.vemv\Scripts\python.exe" (
  "%ROOT%.vemv\Scripts\python.exe" -m hfserve %*
) else if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" -m hfserve %*
) else (
  python -m hfserve %*
)
endlocal
