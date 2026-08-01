@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_qwen_npu_device.ps1" %*
exit /b %ERRORLEVEL%
