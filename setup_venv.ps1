# setup_venv.ps1 - Windows PowerShell setup script

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  QuantumCore Nexus - Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "✓ $pythonVersion detected" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from python.org" -ForegroundColor Yellow
    exit 1
}

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
pip install numpy scipy

# Install full requirements
Write-Host "Installing full requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install in development mode
Write-Host "Installing QuantumCore Nexus in development mode..." -ForegroundColor Yellow
pip install -e .

# Ask about external modules
$download = Read-Host "Download external modules from GitHub? (y/n)"
if ($download -eq 'y') {
    Write-Host "Downloading external modules..." -ForegroundColor Yellow
    python scripts\download_modules.py
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "✅ Setup complete! Virtual environment is ready." -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment:" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "To run QuantumCore Nexus:" -ForegroundColor White
Write-Host "  python main.py" -ForegroundColor Gray
Write-Host "  OR" -ForegroundColor Gray
Write-Host "  quantumcore-nexus" -ForegroundColor Gray
Write-Host ""
Write-Host "To deactivate the virtual environment:" -ForegroundColor White
Write-Host "  deactivate" -ForegroundColor Gray
Write-Host "================================================" -ForegroundColor Cyan