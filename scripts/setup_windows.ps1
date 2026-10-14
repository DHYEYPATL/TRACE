$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "`n=== AssetWatch AI setup ===" -ForegroundColor Cyan
$pythonSelector = $null
# Prefer stable releases; 3.15 is a pre-release as of this writing.
foreach ($candidate in @('3.13', '3.12', '3.11', '3.14', '3.15')) {
    try {
        $out = & py ("-" + $candidate) --version 2>&1
        if ($LASTEXITCODE -eq 0 -and "$out" -match '64') { $pythonSelector = $candidate; break }
        # py --version doesn't always print arch; fall back to accepting any exit-0
        if ($LASTEXITCODE -eq 0) { $pythonSelector = $candidate; break }
    } catch {}
}
if (-not $pythonSelector) {
    Write-Host "Python 3.11-3.15 (64-bit) was not found. Install 64-bit Python from https://www.python.org/downloads/windows/ with the Python Launcher enabled, then rerun this script." -ForegroundColor Red
    exit 1
}
Write-Host "Using Python $pythonSelector"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py ("-" + $pythonSelector) -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv" }
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

$NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($NvidiaSmi) {
    Write-Host "NVIDIA GPU/driver detected. Installing PyTorch CUDA 12.8 build first…" -ForegroundColor Green
    & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CUDA wheel install failed; installing CPU PyTorch fallback." -ForegroundColor Yellow
        & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw "Could not install PyTorch. Follow https://pytorch.org/get-started/locally/ and retry." }
    }
} else {
    Write-Host "nvidia-smi not found. Installing CPU-capable PyTorch; CUDA acceleration needs a supported NVIDIA GPU and driver." -ForegroundColor Yellow
    & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyTorch. Follow https://pytorch.org/get-started/locally/ and retry." }
}
& $Py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency install failed" }
# Ultralytics declares opencv-python; both OpenCV wheels share the cv2 package path.
# Re-install contrib last so its CSRT tracking bindings are the ones imported at runtime.
& $Py -m pip install --force-reinstall --no-deps opencv-contrib-python==4.11.0.86
if ($LASTEXITCODE -ne 0) { throw "Could not install the OpenCV contrib bindings" }
# Install test runner.
& $Py -m pip install pytest>=8.0
if ($LASTEXITCODE -ne 0) { Write-Host "pytest install failed (non-fatal)." -ForegroundColor Yellow }

Write-Host "`n=== Runtime check ===" -ForegroundColor Cyan
& $Py scripts\check_setup.py
if ($LASTEXITCODE -ne 0) { throw "Runtime validation failed. See the diagnostic message above." }

Write-Host "`n=== Offline unit tests ===" -ForegroundColor Cyan
& $Py -m pytest tests\ -v
if ($LASTEXITCODE -ne 0) { Write-Host "One or more unit tests failed. Check output above." -ForegroundColor Yellow }

Write-Host "`nSetup finished. Start the app with .\Run-AssetWatch.bat" -ForegroundColor Green
