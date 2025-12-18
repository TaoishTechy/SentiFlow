#!/bin/bash
# setup_env.sh - Automated virtual environment setup for SentiFlow

echo "================================================"
echo "  SentiFlow AGI Framework - Environment Setup"
echo "================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✓ Python $PYTHON_VERSION detected"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install base requirements
echo "Installing base requirements..."
pip install numpy scipy

# Check for optional dependencies
echo "Checking for optional dependencies..."
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    echo "Detected Ubuntu/Debian system"
    sudo apt-get update
    sudo apt-get install -y python3-tk
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    echo "Detected CentOS/RHEL system"
    sudo yum install -y python3-tkinter
elif command -v brew &> /dev/null; then
    # macOS
    echo "Detected macOS system"
    brew install python-tk
fi

# Install full requirements
if [ -f "requirements.txt" ]; then
    echo "Installing full requirements..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found. Installing core packages..."
    pip install numpy scipy matplotlib psutil pyyaml requests tqdm pandas pytest
fi

# Install SentiFlow in development mode
echo "Installing SentiFlow in development mode..."
pip install -e .

# Download external modules (optional)
if [ -f "scripts/download_modules.py" ]; then
    read -p "Download external quantum modules? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading external modules..."
        python scripts/download_modules.py
    fi
else
    echo "ℹ️  External module script not found. Skipping download."
fi

echo "================================================"
echo "✅ SentiFlow setup complete! Virtual environment is ready."
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests (recommended first step):"
echo "  python examples/qubit_test_32.py"
echo "  OR"
echo "  pytest tests/"
echo ""
echo "Available CLI commands:"
echo "  sentiflow    - Main SentiFlow interface"
echo "  qnvm         - Quantum Network Virtual Machine"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo "================================================"
