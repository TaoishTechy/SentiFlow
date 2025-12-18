"""
Quantum Neuro Virtual Machine (QNVM) v5.1
Main package exports for QNVM framework
"""

from .config import QNVMConfig
from .core import QNVM
from .benchmark import QuantumBenchmark

__version__ = "5.1.0"
__all__ = ['QNVM', 'QNVMConfig', 'QuantumBenchmark']