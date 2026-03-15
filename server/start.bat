@echo off
echo Starting Pain Gauge Backend...
echo.

cd /d "%~dp0"

:: Kill any leftover processes
taskkill /F /IM cloudflared.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1
timeout /t 2 /noq >nul

:: Start the Node server
start "Pain Gauge Server" cmd /k "node index.js"

:: Wait for server to be ready
echo Waiting for server to start...
timeout /t 5 /noq >nul

:: Verify server is running
curl -s http://localhost:3000/api/health >nul 2>&1
if errorlevel 1 (
    echo Server not ready, waiting longer...
    timeout /t 5 /noq >nul
)

:: Start the Cloudflare Tunnel
echo Starting Cloudflare Tunnel...
echo Your backend will be available at: https://paingauge.sidetalk.io
echo.
echo Press Ctrl+C to stop the tunnel.
cloudflared tunnel run paingauge
