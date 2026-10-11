@echo off
echo ============================================================
echo   Starting TRACE: Trusted Real-time Asset Confidence Engine
echo   Mode: Synthetic Branch Simulator (Software-Only Demo)
echo ============================================================
start "" http://localhost:8000
python -m trace.main --config config/demo.yaml --port 8000
pause
