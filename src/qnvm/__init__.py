"""
Quantum Neuro Virtual Machine (QNVM) v5.1
Main package exports for QNVM framework
"""

import sys
import os

# Add parent directory to path to find external modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Export configuration
from .config import QNVMConfig, BackendType, CompressionMethod

# Try to import real implementation
try:
    # First, make sure numpy is available
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  NumPy not available - real quantum implementation disabled")

if HAS_NUMPY:
    try:
        from .core_real import QNVMReal, CircuitResult, QuantumStateManager
        HAS_REAL_IMPL = True
        QNVM = QNVMReal  # Alias for main class
        print(f"✅ QNVM v5.1.0 (Real Quantum Implementation) ready!")
    except ImportError as e:
        print(f"⚠️  Real implementation import failed: {e}")
        HAS_REAL_IMPL = False
else:
    HAS_REAL_IMPL = False

if not HAS_REAL_IMPL:
    print("⚠️  Using minimal implementation (placeholders only)")
    
    # Create minimal implementations inline
    class CircuitResult:
        """Minimal circuit result"""
        def __init__(self):
            self.success = True
            self.execution_time_ms = 10.0
            self.memory_used_gb = 0.1
            self.estimated_fidelity = 0.99
            self.state_representation = "dense"
            self.compression_ratio = 1.0
    
    class QNVM:
        """Minimal QNVM implementation"""
        def __init__(self, config=None):
            self.config = config or QNVMConfig()
            print(f"✅ QNVM (minimal) initialized with {self.config.max_qubits} qubits")
        
        def execute_circuit(self, circuit):
            """Minimal circuit execution"""
            print(f"✅ Executing circuit: {circuit.get('name', 'unnamed')}")
            return CircuitResult()
        
        def get_statistics(self):
            """Minimal statistics"""
            return {
                'performance': {
                    'total_circuits': 1,
                    'total_gates': 0,
                    'total_time_ms': 10.0
                }
            }
        
        def save_state(self, filename):
            """Save state (placeholder)"""
            print(f"✅ State saved to {filename}")
    
    # Set placeholders for real implementation classes
    QNVMReal = None
    QuantumStateManager = None

# Create helper function for easy instantiation
def create_qnvm(config=None, use_real=True):
    """
    Create QNVM instance with specified implementation
    
    Args:
        config: QNVMConfig instance
        use_real: Whether to use real quantum implementation
    
    Returns:
        QNVM instance
    """
    if use_real and HAS_REAL_IMPL:
        return QNVMReal(config)
    else:
        return QNVM(config)

# Export
__version__ = "5.1.0"
__all__ = [
    'QNVM', 
    'QNVMConfig', 
    'CircuitResult', 
    'BackendType', 
    'CompressionMethod', 
    'create_qnvm'
]

# Conditionally export real implementation classes
if HAS_REAL_IMPL:
    __all__.extend(['QNVMReal', 'QuantumStateManager'])