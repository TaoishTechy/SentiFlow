# Contributing to SentiFlow

Thank you for your interest in contributing to SentiFlow! This document provides guidelines and instructions for contributing to our quantum-enhanced sentiment analysis framework.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. By participating in this project, you agree to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

Before you begin contributing, please:

1. **Read the documentation**: Familiarize yourself with the [README.md](README.md) and project structure
2. **Check existing issues**: Browse [open issues](https://github.com/TaoishTechy/SentiFlow/issues) to see if your idea or bug has already been reported
3. **Join discussions**: Participate in [discussions](https://github.com/TaoishTechy/SentiFlow/discussions) to understand project direction

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip package manager
- Virtual environment tool (recommended)

### Setup Steps

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/SentiFlow.git
   cd SentiFlow
   ```

2. **Create a virtual environment**
   ```bash
   # On Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download quantum modules**
   ```bash
   python scripts/download_modules.py
   ```

5. **Run verification tests**
   ```bash
   python verify_installation.py
   ```

6. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Development Environment

- **IDE**: We recommend VS Code or PyCharm with Python extensions
- **Linting**: Use `flake8` or `pylint` for code quality
- **Formatting**: Use `black` for consistent code formatting
- **Type Checking**: Use `mypy` for static type checking

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Help us improve stability and reliability
- **New features**: Propose and implement enhancements
- **Documentation**: Improve guides, examples, and API docs
- **Tests**: Expand test coverage
- **Performance improvements**: Optimize existing code
- **Quantum module development**: Create new quantum-inspired components
- **Examples**: Add usage examples and tutorials

### Contribution Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or for bug fixes
   git checkout -b fix/issue-description
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow our coding standards
   - Add tests for new functionality

3. **Test your changes**
   ```bash
   pytest tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add quantum emotion detection layer"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use descriptive variable and function names

### Code Organization

```python
# Standard library imports
import os
import sys

# Third-party imports
import numpy as np
from typing import Optional, List

# Local imports
from src.sentiflow import SentiFlowAnalyzer
```

### Documentation Standards

- Use docstrings for all public modules, classes, and functions
- Follow Google-style docstring format:

```python
def analyze_sentiment(text: str, quantum_layers: int = 3) -> Dict[str, Any]:
    """Analyzes sentiment using quantum-enhanced processing.
    
    Args:
        text: Input text to analyze
        quantum_layers: Number of quantum processing layers to use
        
    Returns:
        Dictionary containing sentiment scores and metadata
        
    Raises:
        ValueError: If text is empty or quantum_layers is invalid
        
    Example:
        >>> result = analyze_sentiment("Great product!", quantum_layers=2)
        >>> print(result['sentiment'])
        'positive'
    """
    pass
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional

def process_batch(texts: List[str], 
                  batch_size: Optional[int] = None) -> Dict[str, float]:
    """Process multiple texts in batches."""
    pass
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix (e.g., `test_sentiflow.py`)
- Use descriptive test function names

```python
import pytest
from src.sentiflow import SentiFlowAnalyzer

def test_basic_sentiment_analysis():
    """Test basic sentiment analysis functionality."""
    analyzer = SentiFlowAnalyzer()
    result = analyzer.analyze("I love this!")
    assert result.sentiment == "positive"
    assert result.confidence > 0.8

def test_quantum_layer_processing():
    """Test quantum layer enhancement."""
    analyzer = SentiFlowAnalyzer()
    result = analyzer.deep_analyze("Complex emotions", quantum_layers=3)
    assert hasattr(result, 'quantum_entanglement')
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_sentiflow.py

# Run specific test function
pytest tests/test_sentiflow.py::test_basic_sentiment_analysis
```

### Test Coverage

- Aim for at least 80% code coverage
- Focus on critical paths and edge cases
- Include integration tests for quantum modules

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Examples

```bash
feat(quantum-core): add new emotion entanglement algorithm

Implement quantum entanglement for more accurate emotion detection
in multi-layered sentiment analysis.

Closes #123
```

```bash
fix(cognition): resolve memory leak in cognitive processing

The cognitive layer was accumulating state across batches.
Added proper cleanup in the process_batch method.

Fixes #456
```

## Pull Request Process

### Before Submitting

1. **Update documentation** if you've changed APIs
2. **Add tests** for new functionality
3. **Run the full test suite** and ensure all tests pass
4. **Update the README.md** if necessary
5. **Check code quality** with linters

### PR Template

When opening a PR, include:

- **Description**: Clear explanation of changes
- **Motivation**: Why is this change needed?
- **Testing**: How did you test the changes?
- **Screenshots**: If applicable (for UI changes)
- **Related Issues**: Link to related issues

### Review Process

1. A maintainer will review your PR within 3-5 business days
2. Address any requested changes
3. Once approved, a maintainer will merge your PR
4. Your contribution will be included in the next release!

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow guidelines
- [ ] Branch is up to date with main
- [ ] No merge conflicts

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal code to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, OS, relevant dependencies
- **Error messages**: Full stack traces if available

Example:

```markdown
**Bug Description**
Quantum layer processing fails with large batch sizes

**Steps to Reproduce**
1. Initialize analyzer with quantum_layers=5
2. Process batch of 1000+ texts
3. Observe memory error

**Environment**
- Python 3.9.7
- SentiFlow version: 0.1.0
- OS: Ubuntu 20.04

**Error Message**
```
MemoryError: Unable to allocate 2.3 GiB for quantum state
```
```

### Feature Requests

For feature requests, provide:

- **Use case**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Additional context**: Any relevant examples or references

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas

### Getting Help

- Check the [README.md](README.md) first
- Search [existing issues](https://github.com/TaoishTechy/SentiFlow/issues)
- Ask in [discussions](https://github.com/TaoishTechy/SentiFlow/discussions)

### Recognition

Contributors will be:

- Listed in our [CONTRIBUTORS.md](CONTRIBUTORS.md) file
- Mentioned in release notes
- Credited in the project documentation

## License

By contributing to SentiFlow, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to SentiFlow!** Your efforts help advance quantum-enhanced sentiment analysis and cognitive computing research.

For questions about contributing, please open a discussion or contact the maintainers.
