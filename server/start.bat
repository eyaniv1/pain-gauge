@echo off
echo ========================================
echo   Pain Gauge - Starting All Services
echo ========================================
echo.

cd /d "%~dp0"

:: Kill any leftover processes
taskkill /F /IM cloudflared.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1
timeout /t 2 /noq >nul

:: Start the Node server
echo [1/3] Starting Node.js backend...
start "Pain Gauge Server" cmd /k "node index.js"

:: Start the Python inference service
echo [2/3] Starting AI Inference Service...
set INFERENCE_DIR=%~dp0..\inference
if exist "%INFERENCE_DIR%\venv\Scripts\activate.bat" (
    start "AI Inference" cmd /k "cd /d %INFERENCE_DIR% && venv\Scripts\activate && python main.py"
) else (
    echo WARNING: Python venv not found at %INFERENCE_DIR%\venv
    echo AI Inference Service will NOT start. PSPI fallback will be used.
    echo To set up: cd inference ^&^& python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
)

:: Wait for services to be ready
echo.
echo Waiting for services to start...
timeout /t 6 /noq >nul

:: Verify Node.js
curl -s http://localhost:3000/api/health >nul 2>&1
if errorlevel 1 (
    echo WARNING: Node.js server not ready, waiting longer...
    timeout /t 5 /noq >nul
) else (
    echo   Node.js backend: OK
)

:: Verify Python inference
curl -s http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    echo   AI Inference:    NOT RUNNING (will use PSPI fallback)
) else (
    echo   AI Inference:    OK
)

echo.

:: Start the Cloudflare Tunnel
echo [3/3] Starting Cloudflare Tunnel...
echo Your backend will be available at: https://paingauge.sidetalk.io
echo.
echo Press Ctrl+C to stop the tunnel.
cloudflared tunnel run paingauge
