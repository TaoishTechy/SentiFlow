"""
Qubit System Implementation
Optimized for 2-level quantum systems
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .unified_quantum_system import UnifiedQuantumSystem, QuantumSystemConfig, QuantumGate

class QubitSystem(UnifiedQuantumSystem):
    """Optimized 2-level quantum system"""
    
    # Standard single-qubit gates
    PAULI_X = QuantumGate(np.array([[0, 1], [1, 0]]), "X")
    PAULI_Y = QuantumGate(np.array([[0, -1j], [1j, 0]]), "Y")
    PAULI_Z = QuantumGate(np.array([[1, 0], [0, -1]]), "Z")
    HADAMARD = QuantumGate(np.array([[1, 1], [1, -1]]) / np.sqrt(2), "H")
    
    def __init__(self, num_qubits: int, **kwargs):
        config = QuantumSystemConfig(
            num_subsystems=num_qubits,
            dimensions=2,
            **kwargs
        )
        super().__init__(config)
        
    def _initialize_state(self):
        """Initialize to |0...0⟩ state"""
        dim = 2 ** self.config.num_subsystems
        self._state = np.zeros(dim, dtype=np.complex128)
        self._state[0] = 1.0
        
    def apply_gate(self, gate: QuantumGate, targets: List[int]) -> bool:
        """Apply gate to specified qubits"""
        # Validate targets
        for target in targets:
            if target >= self.config.num_subsystems:
                raise ValueError(f"Target qubit {target} out of range")
        
        # Apply gate
        if len(targets) == 1:
            self._apply_single_qubit_gate(gate, targets[0])
        elif len(targets) == 2:
            self._apply_two_qubit_gate(gate, targets)
        else:
            self._apply_general_gate(gate, targets)
        
        self._gate_history.append(gate)
        return True
    
    def _apply_single_qubit_gate(self, gate: QuantumGate, target: int):
        """Apply single-qubit gate using tensor product method"""
        # Create full operator
        operators = [np.eye(2) for _ in range(self.config.num_subsystems)]
        operators[target] = gate.matrix
        
        # Build full operator
        full_op = operators[0]
        for op in operators[1:]:
            full_op = np.kron(full_op, op)
        
        # Apply to state
        self._state = full_op @ self._state
    
    def _apply_two_qubit_gate(self, gate: QuantumGate, targets: List[int]):
        """Apply two-qubit gate"""
        # For CNOT-like gates
        if gate.matrix.shape == (4, 4):
            # Reshape state to matrix form for the two qubits
            state_matrix = self._state.reshape((-1, 4))
            # Apply gate to each 4-dimensional subspace
            transformed = (gate.matrix @ state_matrix.T).T
            self._state = transformed.reshape(-1)
        else:
            # General case: use full operator construction
            self._apply_general_gate(gate, targets)
    
    def _apply_general_gate(self, gate: QuantumGate, targets: List[int]):
        """Apply gate to multiple qubits using full operator construction"""
        # Sort targets
        targets = sorted(targets)
        
        # Build operator for non-target qubits (identity)
        full_dim = 2 ** self.config.num_subsystems
        gate_dim = 2 ** len(targets)
        
        if gate.matrix.shape[0] != gate_dim:
            raise ValueError(f"Gate dimension {gate.matrix.shape[0]} doesn't match target count {len(targets)}")
        
        # Create permutation to bring target qubits to front
        # This is simplified - in production, use more efficient methods
        full_op = np.eye(full_dim, dtype=np.complex128)
        
        # For each basis state, apply gate to target qubits
        for i in range(full_dim):
            # Extract target qubit states
            target_state = 0
            for j, target in enumerate(targets):
                bit = (i >> (self.config.num_subsystems - 1 - target)) & 1
                target_state |= (bit << (len(targets) - 1 - j))
            
            # Apply gate to target state
            new_target_state = gate.matrix[:, target_state]
            
            # Update full state
            for k, amplitude in enumerate(new_target_state):
                if abs(amplitude) > 1e-12:
                    # Construct new index with updated target bits
                    new_index = i
                    for j, target in enumerate(targets):
                        bit = (k >> (len(targets) - 1 - j)) & 1
                        if bit:
                            new_index |= (1 << (self.config.num_subsystems - 1 - target))
                        else:
                            new_index &= ~(1 << (self.config.num_subsystems - 1 - target))
                    
                    full_op[new_index, i] = amplitude
        
        self._state = full_op @ self._state
    
    def measure(self, basis: Optional[np.ndarray] = None, 
                repetitions: int = 1000) -> Dict[str, Any]:
        """Measure in computational basis"""
        probabilities = np.abs(self._state) ** 2
        
        # Generate measurements
        outcomes = np.random.choice(len(probabilities), size=repetitions, p=probabilities)
        
        # Count results
        counts = {}
        for outcome in outcomes:
            # Convert to binary string
            bitstring = format(outcome, f'0{self.config.num_subsystems}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        # Normalize
        total = sum(counts.values())
        probabilities_measured = {k: v/total for k, v in counts.items()}
        
        result = {
            "counts": counts,
            "probabilities": probabilities_measured,
            "theoretical_probabilities": {format(i, f'0{self.config.num_subsystems}b'): float(prob) 
                                          for i, prob in enumerate(probabilities)},
            "repetitions": repetitions,
            "basis": "computational"
        }
        
        self._measurement_history.append(result)
        return result
    
    def create_bell_state(self) -> 'QubitSystem':
        """Create Bell state between first two qubits"""
        self.apply_gate(self.HADAMARD, [0])
        
        # Create CNOT gate
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        cnot_gate = QuantumGate(cnot_matrix, "CNOT")
        self.apply_gate(cnot_gate, [0, 1])
        return self
    
    def create_ghz_state(self) -> 'QubitSystem':
        """Create GHZ state for all qubits"""
        self.apply_gate(self.HADAMARD, [0])
        for i in range(1, self.config.num_subsystems):
            # Create CNOT from qubit 0 to qubit i
            cnot_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ])
            cnot_gate = QuantumGate(cnot_matrix, f"CNOT_0_{i}")
            self.apply_gate(cnot_gate, [0, i])
        return self