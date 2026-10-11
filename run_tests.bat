@echo off
echo ============================================================
echo   Running TRACE Automated Invariant & Security Test Suite
echo ============================================================
python -m pytest -v tests/
pause
