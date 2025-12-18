"""
Minimal QNVM implementation for compatibility
"""

from .config import QNVMConfig

class QNVM:
    """Minimal QNVM class for compatibility"""
    def __init__(self, config=None):
        self.config = config or QNVMConfig()
    
    def execute_circuit(self, circuit):
        """Minimal circuit execution"""
        return type('Result', (), {
            'success': True,
            'execution_time_ms': 10.0,
            'memory_used_gb': 0.1,
            'estimated_fidelity': 0.99,
            'state_representation': 'dense',
            'compression_ratio': 1.0
        })()
    
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
        pass

class CircuitResult:
    """Minimal result class"""
    pass