# src/quantum_core_nexus/core/qudit_system.py
"""
Qudit System Implementation
Multi-level quantum systems with dimension d > 2
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from .unified_quantum_system import UnifiedQuantumSystem, QuantumSystemConfig, QuantumGate

class QuditSystem(UnifiedQuantumSystem):
    """Multi-level quantum system with dimension d > 2"""
    
    def __init__(self, num_qudits: int, dimension: int = 3, **kwargs):
        """
        Initialize a qudit system
        
        Args:
            num_qudits: Number of qudits
            dimension: Dimension of each qudit (must be > 2)
            **kwargs: Additional configuration parameters
        """
        if dimension <= 2:
            raise ValueError(f"Qudits must have dimension > 2. Got d={dimension}")
        
        config = QuantumSystemConfig(
            num_subsystems=num_qudits,
            dimensions=dimension,
            **kwargs
        )
        super().__init__(config)
        self.dimension = dimension
        
        # Initialize generalized Pauli basis
        self._generalized_pauli = self._initialize_generalized_basis()
        
    def _initialize_state(self):
        """Initialize qudit state to |0...0⟩"""
        dim = self.hilbert_dimension
        self._state = np.zeros(dim, dtype=np.complex128)
        self._state[0] = 1.0  # |0...0⟩
        
    def _initialize_generalized_basis(self) -> Dict[str, np.ndarray]:
        """Initialize generalized Pauli operators for qudits"""
        d = self.dimension
        
        # Create phase operator
        omega = np.exp(2j * np.pi / d)
        Z = np.diag([omega**k for k in range(d)])
        
        # Create shift operator
        X = np.zeros((d, d), dtype=np.complex128)
        for i in range(d):
            X[i, (i + 1) % d] = 1.0
        
        return {
            'X': X,      # Generalized X (shift)
            'Z': Z,      # Generalized Z (phase)
            'omega': omega  # Primitive d-th root of unity
        }
    
    def apply_gate(self, gate: QuantumGate, targets: List[int]) -> bool:
        """
        Apply gate to specified qudits
        
        Args:
            gate: Quantum gate to apply
            targets: List of target qudit indices
            
        Returns:
            True if successful, False otherwise
        """
        # Validate targets
        for target in targets:
            if target >= self.config.num_subsystems:
                raise ValueError(f"Target qudit {target} out of range")
        
        # Check gate dimension matches target count
        expected_dim = self.dimension ** len(targets)
        if gate.matrix.shape[0] != expected_dim:
            raise ValueError(
                f"Gate dimension {gate.matrix.shape[0]} doesn't match "
                f"target count {len(targets)} (expected {expected_dim})"
            )
        
        # Apply gate using tensor product method
        self._apply_general_gate(gate, targets)
        self._gate_history.append(gate)
        return True
    
    def _apply_general_gate(self, gate: QuantumGate, targets: List[int]):
        """Apply gate to multiple qudits using full operator construction"""
        # Sort targets for consistent ordering
        targets = sorted(targets)
        num_targets = len(targets)
        
        # Create full operator
        full_op = np.eye(self.hilbert_dimension, dtype=np.complex128)
        
        # For each basis state, apply gate to target qudits
        for i in range(self.hilbert_dimension):
            # Extract target qudit states
            target_state = 0
            for j, target in enumerate(targets):
                # Get the value of this qudit in the overall index i
                # This requires converting between mixed-base representation
                qudit_value = self._get_qudit_value(i, target)
                # Accumulate in base-d representation
                target_state += qudit_value * (self.dimension ** (num_targets - 1 - j))
            
            # Apply gate to target state
            new_target_state = gate.matrix[:, target_state]
            
            # Update full state
            for k, amplitude in enumerate(new_target_state):
                if abs(amplitude) > 1e-12:
                    # Construct new index with updated target qudits
                    new_index = self._set_qudit_values(i, targets, k)
                    full_op[new_index, i] = amplitude
        
        self._state = full_op @ self._state
    
    def _get_qudit_value(self, index: int, qudit: int) -> int:
        """
        Get the value (0 to d-1) of a specific qudit in the overall index
        
        Args:
            index: Overall state index
            qudit: Which qudit to get
            
        Returns:
            Value of the specified qudit (0 to d-1)
        """
        # Convert index to mixed-base representation
        remaining = index
        for q in range(self.config.num_subsystems - 1, -1, -1):
            value = remaining % self.dimension
            if q == qudit:
                return value
            remaining //= self.dimension
        return 0
    
    def _set_qudit_values(self, index: int, qudits: List[int], 
                         target_state: int) -> int:
        """
        Set values for multiple qudits and return new index
        
        Args:
            index: Original state index
            qudits: List of qudit indices to modify
            target_state: New values encoded in base-d
            
        Returns:
            New state index with modified qudits
        """
        # Convert target_state to individual qudit values
        num_targets = len(qudits)
        target_values = []
        temp = target_state
        for _ in range(num_targets):
            target_values.append(temp % self.dimension)
            temp //= self.dimension
        
        # Reverse because we extracted from least significant digit
        target_values.reverse()
        
        # Build new index
        new_index = 0
        multiplier = 1
        
        for q in range(self.config.num_subsystems):
            if q in qudits:
                # Use new value from target_values
                qudit_idx = qudits.index(q)
                value = target_values[qudit_idx]
            else:
                # Use original value
                value = self._get_qudit_value(index, q)
            
            new_index += value * multiplier
            multiplier *= self.dimension
        
        return new_index
    
    def measure(self, basis: Optional[np.ndarray] = None, 
                repetitions: int = 1000) -> Dict[str, Any]:
        """Measure in computational basis"""
        # Calculate probabilities
        probabilities = np.abs(self._state) ** 2
        
        # Ensure probabilities sum to 1 (within tolerance)
        prob_sum = np.sum(probabilities)
        if abs(prob_sum - 1.0) > 1e-10:
            probabilities = probabilities / prob_sum
        
        # Generate measurements
        outcomes = np.random.choice(len(probabilities), size=repetitions, p=probabilities)
        
        # Count results
        counts = {}
        for outcome in outcomes:
            # Convert to base-d string representation
            state_str = self._index_to_string(outcome)
            counts[state_str] = counts.get(state_str, 0) + 1
        
        # Normalize
        total = sum(counts.values())
        probabilities_measured = {k: v/total for k, v in counts.items()}
        
        result = {
            "counts": counts,
            "probabilities": probabilities_measured,
            "theoretical_probabilities": {
                self._index_to_string(i): float(prob) 
                for i, prob in enumerate(probabilities) if prob > 1e-12
            },
            "repetitions": repetitions,
            "basis": "computational",
            "dimension": self.dimension
        }
        
        self._measurement_history.append(result)
        return result
    
    def _index_to_string(self, index: int) -> str:
        """Convert index to string representation in base-d"""
        if index == 0:
            return "0" * self.config.num_subsystems
        
        digits = []
        remaining = index
        for _ in range(self.config.num_subsystems):
            digits.append(str(remaining % self.dimension))
            remaining //= self.dimension
        
        # Reverse and join
        return ''.join(reversed(digits))
    
    def apply_generalized_hadamard(self, target: int) -> bool:
        """Apply generalized Hadamard (QFT on single qudit)"""
        d = self.dimension
        omega = np.exp(2j * np.pi / d)
        
        # Create generalized Hadamard / QFT matrix
        matrix = np.zeros((d, d), dtype=np.complex128)
        for j in range(d):
            for k in range(d):
                matrix[j, k] = omega ** (j * k) / np.sqrt(d)
        
        gate = QuantumGate(matrix, f"Generalized_Hadamard_d{d}")
        return self.apply_gate(gate, [target])
    
    def apply_controlled_increment(self, control: int, target: int) -> bool:
        """Apply controlled increment gate (CINC)"""
        d = self.dimension
        
        # CINC: If control = d-1, increment target by 1 mod d
        matrix_dim = d * d
        matrix = np.eye(matrix_dim, dtype=np.complex128)
        
        for c in range(d):  # Control values
            for t in range(d):  # Target values
                idx_from = c * d + t
                if c == d - 1:  # Only increment when control is maximal
                    idx_to = c * d + ((t + 1) % d)
                else:
                    idx_to = idx_from
                matrix[idx_to, idx_from] = 1.0
        
        gate = QuantumGate(matrix, f"CINC_{control}_{target}")
        return self.apply_gate(gate, [control, target])
    
    def get_level_occupation(self, level: int) -> float:
        """
        Get probability of specific level across all qudits
        
        Args:
            level: Level to measure (0 to d-1)
            
        Returns:
            Total probability of being in specified level
        """
        if not 0 <= level < self.dimension:
            raise ValueError(f"Level must be between 0 and {self.dimension-1}")
        
        total_prob = 0.0
        for i, amplitude in enumerate(self._state):
            # Check if this index has the specified level in any qudit
            # For simplicity, check the first qudit's value
            qudit_value = i % self.dimension
            if qudit_value == level:
                total_prob += abs(amplitude) ** 2
        
        return total_prob
    
    def create_generalized_bell_state(self) -> 'QuditSystem':
        """
        Create generalized Bell state for 2 qudits:
        |Φ⟩ = 1/√d ∑_{k=0}^{d-1} |k⟩⊗|k⟩
        """
        if self.config.num_subsystems != 2:
            raise ValueError("Generalized Bell state requires exactly 2 qudits")
        
        # Apply generalized Hadamard to first qudit
        self.apply_generalized_hadamard(0)
        
        # Apply controlled increment
        self.apply_controlled_increment(0, 1)
        
        return self
    
    def calculate_schmidt_coefficients(self) -> np.ndarray:
        """
        Calculate Schmidt coefficients for bipartite system
        
        Returns:
            Array of Schmidt coefficients (singular values)
        """
        if self.config.num_subsystems != 2:
            raise ValueError("Schmidt decomposition requires exactly 2 subsystems")
        
        # Reshape state vector to matrix
        psi_matrix = self._state.reshape((self.dimension, self.dimension))
        
        # Perform SVD
        U, S, Vh = np.linalg.svd(psi_matrix)
        
        return S