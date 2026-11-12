@echo off
rem Clears the loaded model from GPU memory and stops the hfserve server.
rem Pass -All to stop every hfserve server on this machine.
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%Stop-HFServe.ps1" %*
endlocal
