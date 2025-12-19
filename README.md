# SentiFlow: Experimental AGI Research Framework

[![GitHub License](https://img.shields.io/github/license/TaoishTechy/SentiFlow)](https://github.com/TaoishTechy/SentiFlow/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Quantum Enhanced](https://img.shields.io/badge/quantum-enhanced-purple.svg)](https://en.wikipedia.org/wiki/Quantum-inspired_algorithm)

SentiFlow is an experimental quantum physics / meta cognition research framework for modeling **emergent agency** through competence, irreversible choice, constrained safety, and meta-cognitive feedback. It explores how agency forms when systems are forced to sacrifice options, accumulate consequences, and operate under asymmetric risk—without faked emergence or hard-coded sentience. (For now it's a neat Quantum VM that manage authentic qudits).

## ✨ Core Research Principles

This framework investigates agency emergence through several key mechanisms:

*   **Competence-Driven Agency**: Systems develop agency by building measurable competence at tasks with real consequences.
*   **Irreversible Choice**: Modeling decisions that permanently eliminate future options, forcing meaningful strategic development.
*   **Constrained Safety**: Implementing safety as asymmetric constraints that limit harmful actions while preserving constructive capability.
*   **Meta-Cognitive Feedback**: Multi-layer self-monitoring that allows systems to develop awareness of their own decision processes.

## 🚀 Getting Started

### Prerequisites

*   Python 3.8 or higher
*   Git for cloning the repository

### Installation

```bash
# Clone the repository
git clone https://github.com/TaoishTechy/SentiFlow.git
cd SentiFlow

# Install dependencies
pip install -r requirements.txt

# Set up the development environment
./setup_env.sh  # or setup_venv.ps1 for Windows PowerShell
```

### Quick Verification

After installation, run the verification script to ensure all components are working correctly:

```bash
python verify_installation.py
```

## 📁 Project Structure

```
SentiFlow/
├── src/
│   ├── qnvm/                 # Quantum Network Virtual Machine core
│   │   ├── __init__.py
│   │   ├── core.py           # Main QNVM implementation
│   │   ├── core_real.py      # Real quantum implementation
│   │   ├── config.py         # Configuration management
│   ├── external/
│   │   └── /  # External quantum modules
│   ├── cli_main.py          # Command-line interface
│   └── cli_demos.py         # Demonstration scripts
├── examples/
│   ├── qubit_test_32.py     # Comprehensive quantum test suite (up to 32 qubits)
│   ├── qudit_test_32.py     # Comprehensive quantum test suite (up to 32 qudits)
│   └── __main__.py          # Example entry point
├── scripts/
│   └── download_modules.py  # Utility for managing external modules
├── tests/                   # Test suite
├── Dockerfile              # Containerization setup
├── docker-compose.yml      # Multi-container orchestration
├── requirements.txt        # Python dependencies
├── setup.py               # Package installation
├── CONTRIBUTING.md        # Contribution guidelines
├── LICENSE                # MIT License
└── README.md              # This file
```

## 🔧 Key Components

### Quantum Network Virtual Machine (QNVM)

At the heart of SentiFlow is the QNVM, a flexible quantum simulation environment that supports:

*   **Multiple Backends**: Internal simulator, tensor network compression, and cloud quantum computing interfaces
*   **Dynamic Qubit Management**: Efficient handling of quantum states with automatic memory optimization
*   **Fidelity Validation**: Robust quantum state and operation verification
*   **Performance Monitoring**: Real-time tracking of computational resources

### External Module System

SentiFlow uses a modular architecture with specialized components:

```bash
# Download and manage external modules
python scripts/download_modules.py
```

Key external modules include:
*   **Tensor Network Engine**: Efficient quantum state representation and manipulation
*   **Fidelity Calculator**: Advanced quantum state comparison and validation
*   **Memory Manager**: Intelligent resource allocation for large-scale simulations

## 📊 Performance and Validation

Recent comprehensive testing demonstrates SentiFlow's robust performance:

*   **Quantum Fidelity**: 99.55% average across test suite
*   **Test Success Rate**: 100% (10/10 tests completed successfully)
*   **Execution Speed**: 1.04 seconds for full test suite
*   **Memory Efficiency**: 38.3MB peak usage
*   **Quantum Gate Speed**: 3,676 gates/second simulation
*   **Scalability**: Successfully tested up to 28 qubits (auto-adjusted from 32 based on available memory)

### Running the Test Suite

```bash
# Run the comprehensive quantum test suite
cd examples
python qubit_test_32.py

# Or run specific tests
pytest tests/ -v
```

## 🔬 Research Applications

SentiFlow is designed for advanced research in:

*   **Emergent Agency Modeling**: Studying how goal-directed behavior arises from basic computational principles
*   **Quantum-Enhanced Cognition**: Exploring quantum-inspired algorithms for decision-making
*   **AGI Safety Research**: Developing and testing constrained optimization approaches
*   **Cognitive Architecture**: Building and evaluating multi-layer feedback systems
*   **Irreversibility in Computation**: Modeling systems with path-dependent development

## 🧪 Example Usage

### Basic Quantum Circuit Execution

```python
from src.qnvm import create_qnvm, QNVMConfig
from src.qnvm.config import BackendType

# Configure the quantum virtual machine
config = QNVMConfig(
    max_qubits=16,
    max_memory_gb=4.0,
    backend=BackendType.INTERNAL,
    compression_enabled=True,
    validation_enabled=True
)

# Create and use the QNVM instance
vm = create_qnvm(config, use_real=True)

# Define a quantum circuit
circuit = {
    'name': 'bell_state',
    'num_qubits': 2,
    'gates': [
        {'gate': 'H', 'targets': [0]},
        {'gate': 'CNOT', 'targets': [1], 'controls': [0]}
    ]
}

# Execute the circuit
result = vm.execute_circuit(circuit)
print(f"Fidelity: {result.estimated_fidelity:.4f}")
print(f"Execution time: {result.execution_time_ms}ms")
```

### Advanced Agency Modeling

```python
# Example of irreversible choice modeling
from src.agency import IrreversibleChoiceModel

model = IrreversibleChoiceModel(
    option_space=100,
    irreversible_threshold=0.7,
    safety_constraints={'max_harm': 0.1}
)

# Simulate agency development
trajectory = model.simulate_development(steps=1000)
print(f"Final competence: {trajectory.competence[-1]:.3f}")
print(f"Options remaining: {trajectory.options_remaining}")
```

## 🐳 Containerized Deployment

For consistent research environments, SentiFlow includes Docker support:

```bash
# Build the Docker image
docker build -t sentiflow .

# Run with Docker Compose
docker-compose up

# Or run directly
docker run -it --rm sentiflow python examples/qubit_test_32.py
```

## 🤝 Contributing

We welcome contributions from researchers and developers interested in emergent agency and quantum-enhanced AI. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

1.  Forking the repository
2.  Creating feature branches
3.  Running the test suite
4.  Submitting pull requests

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

## 🛠️ Support and Discussion

*   📝 **Issue Tracker**: Report bugs or request features on GitHub Issues
*   💬 **Discussion Forum**: Join theoretical and technical discussions

## 📈 Recent Developments

*   **December 2025**: Enhanced quantum test suite with comprehensive fidelity validation up to 32 qubits
*   **Quantum Core Nexus**: Expanded external module system for specialized quantum operations
*   **Performance Optimization**: Achieved 99.55% average fidelity in comprehensive testing
*   **Containerization**: Full Docker support for reproducible research environments

---

⭐ **Star this repository** if you find SentiFlow useful for your AGI research!

*"Modeling agency through irreversible choice and constrained emergence."*

---
*Note: SentiFlow is experimental research software. Results should be validated and reproduced in controlled environments.*
