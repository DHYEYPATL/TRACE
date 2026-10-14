$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "Virtual environment not found. Run .\scripts\setup_windows.ps1 first." -ForegroundColor Yellow
    exit 1
}
& $Py app.py
