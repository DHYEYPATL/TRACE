@echo off
echo ============================================================
echo   Starting TRACE: Trusted Real-time Asset Confidence Engine
echo   Mode: Live Webcam (Hardware + Software Demo)
echo ============================================================
start "" http://localhost:8000
python -m trace.main --config config/webcam.yaml --port 8000
pause
