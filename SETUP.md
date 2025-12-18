Based on the updated setup files for SentiFlow, here is a comprehensive `SETUP.md` file that provides clear installation and setup instructions:

```markdown
# SentiFlow Setup Guide

## Overview

SentiFlow is an experimental AGI research framework for modeling emergent agency through quantum-inspired computing. This guide covers installation, environment setup, and verification for all major platforms.

## Prerequisites

### Python Requirements
- **Python 3.8 or higher** (3.10+ recommended)
- **pip** (Python package installer)
- **virtualenv** or **venv** (for isolated environments)

### System Dependencies
- **Linux/macOS**: Standard development tools
- **Windows**: PowerShell 5.1+ or Windows Terminal
- **Memory**: Minimum 4GB RAM (8GB+ recommended for quantum simulations)
- **Disk Space**: ~500MB for installation

## Quick Start

### Automated Setup (Recommended)

#### Linux/macOS
```bash
# Make script executable and run
chmod +x setup_env.sh
./setup_env.sh
```

#### Windows
```powershell
# Run PowerShell script
.\setup_venv.ps1
```

The automated scripts will:
1. Create a Python virtual environment
2. Install all dependencies
3. Set up SentiFlow in development mode
4. Optionally download external quantum modules

### Manual Setup

If you prefer manual control or the automated scripts fail:

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate environment:**
   - **Linux/macOS:** `source venv/bin/activate`
   - **Windows:** `venv\Scripts\activate`

3. **Install SentiFlow:**
   ```bash
   # Upgrade pip first
   pip install --upgrade pip

   # Install with minimal dependencies
   pip install -e .[minimal]

   # Or install with full dependencies
   pip install -e .[full]

   # For development (includes testing tools)
   pip install -e .[dev]
   ```

## Installation Methods

### 1. Minimal Installation (Core functionality)
```bash
pip install -e .[minimal]
```
**Includes:** numpy, scipy, and essential quantum simulation libraries

### 2. Full Installation (Recommended)
```bash
pip install -e .[full]
```
**Includes:** All scientific computing packages, visualization tools, and utilities

### 3. Development Installation
```bash
pip install -e .[dev]
```
**Includes:** Testing frameworks, code quality tools, and development dependencies

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-tk python3-dev build-essential

# Run setup
./setup_env.sh
```

### macOS
```bash
# Ensure Homebrew is installed (if not)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python TK if needed
brew install python-tk

# Run setup
./setup_env.sh
```

### Windows
```powershell
# Enable script execution if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run setup
.\setup_venv.ps1
```

## Verifying Installation

### Quick Test
```python
python -c "import qnvm; print('SentiFlow QNVM loaded successfully')"
```

### Run Comprehensive Test Suite
```bash
# Run the quantum test suite
python examples/qubit_test_32.py

# Or run pytest tests
pytest tests/ -v
```

### Test Quantum Simulation
```bash
# Test basic quantum operations
python -c "
from src.qnvm import create_qnvm, QNVMConfig
config = QNVMConfig(max_qubits=4)
vm = create_qnvm(config)
print('Quantum VM initialized successfully')
"
```

## Using the CLI

After installation, SentiFlow provides command-line interfaces:

```bash
# Main SentiFlow interface
sentiflow --help

# Quantum Network Virtual Machine
qnvm --help

# Run test demos
sentiflow-test
```

## Docker Setup

For containerized deployment:

```bash
# Build Docker image
docker build -t sentiflow .

# Run with Docker Compose
docker-compose up

# Or run directly
docker run -it --rm sentiflow python examples/qubit_test_32.py
```

## Troubleshooting

### Common Issues

1. **Python version issues:**
   ```bash
   # Check Python version
   python --version
   
   # Ensure Python 3.8+
   python3 --version
   ```

2. **Virtual environment activation fails:**
   - **Windows:** Ensure you're using PowerShell, not Command Prompt
   - **Linux/macOS:** Check execute permissions: `chmod +x venv/bin/activate`

3. **Missing dependencies:**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

4. **External module download fails:**
   ```bash
   # Manual download
   python scripts/download_modules.py --force
   ```

### Getting Help

- **Check logs:** Review installation logs in `setup.log`
- **Test individually:** Run each setup step separately
- **Clean install:** Remove `venv/` directory and start fresh
- **Issue tracker:** Report bugs at [GitHub Issues](https://github.com/TaoishTechy/SentiFlow/issues)

## Post-Installation

### First Steps
1. **Verify installation** with the test suite
2. **Explore examples** in the `examples/` directory
3. **Review documentation** in the project README
4. **Try basic quantum circuits** with the QNVM

### Development Workflow
```bash
# Activate environment
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate   # Windows

# Make changes to source code
# Test changes
pytest tests/

# Deactivate when done
deactivate
```

## Performance Notes

Based on recent benchmarks, SentiFlow achieves:
- **99.55% average quantum fidelity** across test suite
- **3,676 gates/second** simulation speed
- **38.3MB peak memory usage** for comprehensive testing
- **Up to 28 qubits** tested successfully (auto-adjusted from 32 based on available memory)

For optimal performance:
- Use **sparse mode** for systems with >16 qubits
- Allocate **minimum 4GB RAM** for quantum simulations
- Enable **parallel processing** where available

## Next Steps

After successful setup:

1. **Run the quantum test suite** to verify performance
2. **Explore the examples/** directory for usage patterns
3. **Review the research documentation** for theoretical background
4. **Check CONTRIBUTING.md** if interested in development
5. **Join the discussion forum** for questions and collaboration

---

*For additional help or to report issues, please visit the [SentiFlow GitHub repository](https://github.com/TaoishTechy/SentiFlow).*
```

This `SETUP.md` provides comprehensive installation instructions that:

1. **Covers all platforms** (Linux, macOS, Windows)
2. **Includes multiple installation methods** (automated scripts, manual setup, Docker)
3. **Provides verification steps** to ensure proper installation
4. **Includes troubleshooting guidance** for common issues
5. **References the actual project structure** and components
6. **Includes performance benchmarks** based on your recent test results

The document is structured to guide users from simple installation to advanced configuration, with clear examples and platform-specific instructions.
