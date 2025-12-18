# setup_venv.ps1 - Automated virtual environment setup for Windows

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  SentiFlow AGI Framework - Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check if Python 3 is installed
$pythonVersion = (python --version 2>&1).ToString()
if (-not $pythonVersion -like "Python 3*") {
    Write-Host "❌ Python 3 is not installed. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

Write-Host "✓ $pythonVersion detected" -ForegroundColor Green

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install base requirements
Write-Host "Installing base requirements..." -ForegroundColor Yellow
python -m pip install numpy scipy

# Install full requirements if requirements.txt exists
if (Test-Path "requirements.txt") {
    Write-Host "Installing full requirements..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
} else {
    Write-Host "⚠️ requirements.txt not found. Installing core packages..." -ForegroundColor Yellow
    python -m pip install numpy scipy matplotlib psutil pyyaml requests tqdm pandas pytest
}

# Install SentiFlow in development mode
Write-Host "Installing SentiFlow in development mode..." -ForegroundColor Yellow
python -m pip install -e .

# Download external modules (optional)
if (Test-Path "scripts/download_modules.py") {
    $response = Read-Host "Download external quantum modules? (y/n)"
    if ($response -eq 'y') {
        Write-Host "Downloading external modules..." -ForegroundColor Yellow
        python scripts/download_modules.py
    }
} else {
    Write-Host "ℹ️ External module script not found. Skipping download." -ForegroundColor Yellow
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "✅ SentiFlow setup complete! Virtual environment is ready." -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment in the future:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To run tests (recommended first step):" -ForegroundColor Yellow
Write-Host "  python examples/qubit_test_32.py" -ForegroundColor White
Write-Host "  OR" -ForegroundColor White
Write-Host "  pytest tests/" -ForegroundColor White
Write-Host ""
Write-Host "Available CLI commands:" -ForegroundColor Yellow
Write-Host "  sentiflow    - Main SentiFlow interface" -ForegroundColor White
Write-Host "  qnvm         - Quantum Network Virtual Machine" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate the virtual environment:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host "================================================" -ForegroundColor Cyan
