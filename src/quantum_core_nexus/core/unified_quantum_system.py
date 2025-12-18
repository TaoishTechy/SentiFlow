"""
Unified Quantum System Base Class
Provides foundation for both qubit and qudit systems with scientific validation
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Union, Optional, Any
import time
import warnings

class QuantumSystemType(Enum):
    QUBIT = 2
    QUDIT = "variable"
    HYBRID = "mixed"

@dataclass
class QuantumSystemConfig:
    """Configuration for any quantum system"""
    num_subsystems: int
    dimensions: Union[int, List[int]]
    representation: str = "dense"
    precision: str = "complex128"
    validation_level: str = "strict"
    
class QuantumGate:
    """Base class for quantum gates"""
    def __init__(self, matrix: np.ndarray, name: str = None):
        self.matrix = matrix
        self.name = name or f"Gate_{id(self)}"
        self._validate_unitarity()
    
    def _validate_unitarity(self, tolerance: float = 1e-10):
        """Validate that the gate is unitary"""
        identity = np.eye(self.matrix.shape[0])
        product = self.matrix.conj().T @ self.matrix
        if not np.allclose(product, identity, atol=tolerance):
            warnings.warn(f"Gate {self.name} may not be unitary")
    
    @property
    def dimension(self) -> int:
        return self.matrix.shape[0]

class UnifiedQuantumSystem(ABC):
    """
    Base class for all quantum systems with scientific validation
    """
    
    def __init__(self, config: QuantumSystemConfig):
        self.config = config
        self._state = None
        self._density_matrix = None
        self._gate_history = []
        self._measurement_history = []
        self._validation_suite = None  # Will be initialized later
        
        # Initialize based on configuration
        self._initialize_state()
        
    @abstractmethod
    def _initialize_state(self):
        """Initialize quantum state based on configuration"""
        pass
    
    @abstractmethod
    def apply_gate(self, gate: QuantumGate, targets: List[int]) -> bool:
        """Apply quantum gate with unitarity validation"""
        pass
    
    @abstractmethod
    def measure(self, basis: Optional[np.ndarray] = None, 
                repetitions: int = 1000) -> Dict[str, Any]:
        """Statistical measurement with Born rule"""
        pass
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state vector"""
        return self._state.copy()
    
    def get_density_matrix(self) -> np.ndarray:
        """Get density matrix representation"""
        if self._density_matrix is None:
            self._density_matrix = np.outer(self._state, self._state.conj())
        return self._density_matrix
    
    def calculate_entropy(self, subsystem: List[int]) -> float:
        """Calculate von Neumann entropy for subsystem"""
        from ..validation.metric_calculator import QuantumMetricCalculator
        rho = self._get_reduced_density_matrix(subsystem)
        return QuantumMetricCalculator.calculate_von_neumann_entropy(rho)
    
    def calculate_coherence(self, basis: str = "computational") -> float:
        """Calculate quantum coherence"""
        from ..validation.metric_calculator import QuantumMetricCalculator
        rho = self.get_density_matrix()
        return QuantumMetricCalculator.calculate_coherence(rho, basis)
    
    def _get_reduced_density_matrix(self, subsystem: List[int]) -> np.ndarray:
        """Calculate reduced density matrix for subsystem"""
        # This is a simplified implementation
        # For full implementation, we need to trace out other subsystems
        if len(subsystem) == self.config.num_subsystems:
            return self.get_density_matrix()
        
        # Placeholder: Return subsystem density matrix
        dim = 2 ** len(subsystem) if isinstance(self.config.dimensions, int) else np.prod([self.config.dimensions[i] for i in subsystem])
        return np.eye(dim) / dim
    
    @property
    def hilbert_dimension(self) -> int:
        """Total dimension of Hilbert space"""
        if isinstance(self.config.dimensions, int):
            return self.config.dimensions ** self.config.num_subsystems
        else:
            dim = 1
            for d in self.config.dimensions:
                dim *= d
            return dim
    
    @property
    def state_norm(self) -> float:
        """State vector norm with tolerance check"""
        if self._state is None:
            return 0.0
        norm = np.linalg.norm(self._state)
        if abs(norm - 1.0) > 1e-12:
            if self.config.validation_level == "strict":
                raise ValueError(f"State norm deviation: {abs(norm - 1.0):.2e}")
            elif self.config.validation_level == "warn":
                warnings.warn(f"State norm deviation: {abs(norm - 1.0):.2e}")
        return norm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system state to dictionary"""
        return {
            "config": self.config.__dict__,
            "state_vector": self._state.tolist() if self._state is not None else None,
            "hilbert_dimension": self.hilbert_dimension,
            "state_norm": self.state_norm,
            "gate_history": [gate.name for gate in self._gate_history]
        }