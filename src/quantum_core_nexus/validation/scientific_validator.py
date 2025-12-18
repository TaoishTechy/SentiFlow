"""
Scientific Validation Engine
Validates quantum principles and calculations
"""

import numpy as np
from typing import Dict, Tuple, List, Any
import time
from scipy import stats
from ..core.unified_quantum_system import UnifiedQuantumSystem

class QuantumPrincipleViolation(Exception):
    """Exception for quantum principle violations"""
    pass

class QuantumValidator:
    """Comprehensive quantum principle validation"""
    
    VALIDATION_TESTS = {
        "unitarity": {
            "function": "validate_unitarity",
            "tolerance": 1e-10,
            "description": "Gate operations must be unitary"
        },
        "born_rule": {
            "function": "validate_born_rule",
            "tolerance": 1e-8,
            "description": "Measurement probabilities sum to 1"
        },
        "state_normalization": {
            "function": "validate_state_normalization",
            "tolerance": 1e-12,
            "description": "State vector norm = 1"
        }
    }
    
    def __init__(self, quantum_system: UnifiedQuantumSystem):
        self.system = quantum_system
        self.violation_log = []
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive quantum validation suite"""
        results = {}
        
        for test_name, test_config in self.VALIDATION_TESTS.items():
            test_func = getattr(self, test_config["function"])
            try:
                passed, message = test_func(test_config["tolerance"])
                results[test_name] = {
                    "passed": passed,
                    "message": message,
                    "timestamp": time.time()
                }
                if not passed:
                    self.log_violation(test_name, message)
            except Exception as e:
                results[test_name] = {
                    "passed": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
                self.log_violation(test_name, f"Test error: {e}")
                
        return results
    
    def validate_unitarity(self, gate_matrix: np.ndarray = None, 
                          tolerance: float = 1e-10) -> Tuple[bool, str]:
        """Validate U†U = I within tolerance"""
        if gate_matrix is None and self.system._gate_history:
            gate_matrix = self.system._gate_history[-1].matrix
        elif gate_matrix is None:
            return True, "No gates to validate"
            
        identity = np.eye(gate_matrix.shape[0])
        product = gate_matrix.conj().T @ gate_matrix
        
        max_deviation = np.max(np.abs(product - identity))
        passed = max_deviation < tolerance
        
        message = (f"Unitarity check: max deviation = {max_deviation:.2e} "
                  f"(tolerance = {tolerance:.0e})")
        
        return passed, message
    
    def validate_born_rule(self, measurements: Dict[str, float] = None, 
                          tolerance: float = 1e-8) -> Tuple[bool, str]:
        """Validate probabilities sum to 1"""
        if measurements is None:
            # Calculate from state vector
            probabilities = np.abs(self.system._state) ** 2
            total_prob = np.sum(probabilities)
        else:
            total_prob = sum(measurements.values())
            
        deviation = abs(total_prob - 1.0)
        passed = deviation < tolerance
        
        message = f"Born rule: sum(prob) = {total_prob:.10f}, deviation = {deviation:.2e}"
        return passed, message
    
    def validate_state_normalization(self, tolerance: float = 1e-12) -> Tuple[bool, str]:
        """Validate state vector normalization"""
        norm = self.system.state_norm
        deviation = abs(norm - 1.0)
        passed = deviation < tolerance
        
        message = f"State normalization: norm = {norm:.12f}, deviation = {deviation:.2e}"
        return passed, message
    
    def log_violation(self, test_name: str, message: str):
        """Log quantum principle violations"""
        violation = {
            "test": test_name,
            "message": message,
            "timestamp": time.time(),
            "state_norm": self.system.state_norm,
            "system_config": self.system.config.__dict__
        }
        self.violation_log.append(violation)
        
        if self.system.config.validation_level == "strict":
            raise QuantumPrincipleViolation(f"{test_name}: {message}")