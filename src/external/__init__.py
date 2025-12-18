# /src/external/__init__.py
"""
External quantum computing modules and utilities.
Provides tensor networks, fidelity calculations, and other quantum tools.
"""

# Tensor network implementations
try:
    from .tensor_network import TensorNetwork, MatrixProductState, TensorTrain
    HAS_TENSOR_NETWORK = True
except ImportError as e:
    print(f"⚠️  TensorNetwork import error: {e}")
    HAS_TENSOR_NETWORK = False
    
    # Define placeholder classes
    class TensorNetwork:
        """Placeholder for tensor network implementation"""
        pass
    
    class MatrixProductState:
        """Placeholder for MPS implementation"""
        pass
    
    class TensorTrain:
        """Placeholder for tensor train implementation"""
        pass

# Fidelity calculations
try:
    from .fidelity_fix import (
        FidelityCalculator,
        StateVerification,
        QuantumMetrics,
        validate_quantum_state,
        calculate_state_fidelity,
        calculate_gate_fidelity
    )
    HAS_FIDELITY = True
except ImportError as e:
    print(f"⚠️  Fidelity module import error: {e}")
    HAS_FIDELITY = False
    
    # Define placeholder classes for fidelity
    class FidelityCalculator:
        """Placeholder for fidelity calculator"""
        @staticmethod
        def calculate_state_fidelity(ideal_state, actual_state):
            return 0.0
    
    class StateVerification:
        """Placeholder for state verification"""
        pass
    
    class QuantumMetrics:
        """Placeholder for quantum metrics"""
        pass

# Quantum memory management
try:
    from .memory_manager import QuantumMemoryManager, MemoryAllocator
    HAS_MEMORY_MANAGER = True
except ImportError as e:
    print(f"⚠️  MemoryManager import error: {e}")
    HAS_MEMORY_MANAGER = False
    
    class QuantumMemoryManager:
        """Placeholder for quantum memory manager"""
        pass
    
    class MemoryAllocator:
        """Placeholder for memory allocator"""
        pass

# Export main classes
__all__ = [
    # Tensor networks
    'TensorNetwork',
    'MatrixProductState',
    'TensorTrain',
    'HAS_TENSOR_NETWORK',
    
    # Fidelity
    'FidelityCalculator',
    'StateVerification',
    'QuantumMetrics',
    'HAS_FIDELITY',
    
    # Memory management
    'QuantumMemoryManager',
    'MemoryAllocator',
    'HAS_MEMORY_MANAGER',
    
    # Helper functions
    'check_dependencies',
    'get_available_features'
]

def check_dependencies():
    """Check which dependencies are available"""
    return {
        'tensor_network': HAS_TENSOR_NETWORK,
        'fidelity': HAS_FIDELITY,
        'memory_manager': HAS_MEMORY_MANAGER
    }

def get_available_features():
    """Get list of available features"""
    features = []
    if HAS_TENSOR_NETWORK:
        features.append('tensor_networks')
    if HAS_FIDELITY:
        features.append('fidelity_calculations')
    if HAS_MEMORY_MANAGER:
        features.append('memory_management')
    return features
