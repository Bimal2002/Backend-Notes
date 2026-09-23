$ErrorActionPreference = 'Stop'

Write-Host "Setting up Fever AI MVP..." -ForegroundColor Cyan

$pythonCmd = $null

if (Get-Command py -ErrorAction SilentlyContinue) {
  # Prefer versions with the best binary wheel coverage.
  try { py -3.12 --version | Out-Null; $pythonCmd = 'py -3.12' } catch {}
  if (-not $pythonCmd) { try { py -3.11 --version | Out-Null; $pythonCmd = 'py -3.11' } catch {} }
}

if (-not $pythonCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
  $pythonCmd = 'python'
}

if (-not $pythonCmd) {
  Write-Host "Python not found." -ForegroundColor Yellow
  Write-Host "Install Python 3.11+ from https://www.python.org/downloads/ and re-run this script." -ForegroundColor Yellow
  Write-Host "If you installed from Microsoft Store, ensure App Execution Aliases don't override python." -ForegroundColor Yellow
  exit 1
}

Write-Host "Using: $pythonCmd" -ForegroundColor Green

$pyVer = & $pythonExe @pythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Detected Python version: $pyVer" -ForegroundColor Green

if ($pyVer -match '^(\d+)\.(\d+)$') {
  $maj = [int]$Matches[1]
  $min = [int]$Matches[2]
  if ($maj -eq 3 -and $min -ge 13) {
    Write-Host "Using lite mode (UI-only) for Python 3.13+." -ForegroundColor Yellow
    Write-Host "This avoids FastAPI/Pydantic and heavy scientific wheels." -ForegroundColor Yellow

    Set-Location (Split-Path $PSScriptRoot -Parent)
    & $pythonExe @pythonArgs -m venv .venv
    . .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip setuptools wheel
    pip install -r requirements-lite.txt

    & $pythonExe @pythonArgs -m ml.generate_synthetic_data --out data\training_data.csv --rows 800
    & $pythonExe @pythonArgs -m ml.train_lite_model --data data\training_data.csv --out ml\model.json

    Write-Host "Done. Run: streamlit run ui\app.py" -ForegroundColor Green
    exit 0
  }
}

Set-Location (Split-Path $PSScriptRoot -Parent)

$pythonExe = $pythonCmd.Split(' ')[0]
$pythonArgs = @($pythonCmd.Split(' ') | Select-Object -Skip 1)

& $pythonExe @pythonArgs -m venv .venv

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

& $pythonExe @pythonArgs -m ml.generate_synthetic_data --out data\training_data.csv --rows 800
& $pythonExe @pythonArgs -m ml.train_model --data data\training_data.csv --out ml\model.joblib

Write-Host "Done. Run: streamlit run ui\app.py" -ForegroundColor Green
