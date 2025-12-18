# src/quantum_core_nexus/demonstrations/qudit_demos.py
"""
Qudit Demonstrations
Advanced demonstrations for multi-level quantum systems
"""

import numpy as np
from typing import Dict, List, Any, Optional
import time
from ..core.qudit_system import QuditSystem
from ..core.unified_quantum_system import QuantumGate
from ..validation.metric_calculator import QuantumMetricCalculator
from ..validation.scientific_validator import QuantumValidator

class QuditDemonstrationSuite:
    """Advanced qudit demonstrations"""
    
    def __init__(self, dimension: int = 3):
        """
        Initialize qudit demonstration suite
        
        Args:
            dimension: Dimension of qudits (d > 2)
        """
        if dimension <= 2:
            raise ValueError(f"Qudits must have dimension > 2. Got d={dimension}")
        
        self.dimension = dimension
        self.results = {}
        
    def run_all_demos(self, max_qudits: int = 3) -> Dict[str, Any]:
        """
        Run all qudit demonstrations
        
        Args:
            max_qudits: Maximum number of qudits to test
            
        Returns:
            Dictionary of demonstration results
        """
        all_results = {}
        
        print(f"\n{'='*70}")
        print(f"RUNNING QUDIT DEMONSTRATIONS (d={self.dimension})")
        print('='*70)
        
        # Test different numbers of qudits
        for n in [2, min(3, max_qudits), min(4, max_qudits)]:
            if n <= max_qudits:
                print(f"\n▶ Testing with {n} qudit{'s' if n > 1 else ''}")
                
                # Run individual demos
                bell_results = self.demo_generalized_bell_states(n)
                superposition_results = self.demo_superposition_states(n)
                qft_results = self.demo_quantum_fourier_transform_qudit(n)
                teleport_results = self.demo_qudit_teleportation() if n >= 3 else {}
                
                all_results[f"{n}_qudits"] = {
                    "generalized_bell": bell_results,
                    "superposition": superposition_results,
                    "qft": qft_results,
                    "teleportation": teleport_results
                }
        
        self.results = all_results
        return all_results
    
    def demo_generalized_bell_states(self, num_qudits: int = 2) -> Dict[str, Any]:
        """Create and analyze generalized Bell states"""
        print(f"\n1. Generalized Bell State (d={self.dimension}, n={num_qudits})")
        print("-" * 40)
        
        start_time = time.time()
        
        if num_qudits != 2:
            print(f"  Note: Bell state demonstration optimized for 2 qudits")
        
        # Create system
        system = QuditSystem(num_qudits, self.dimension, validation_level="warn")
        
        # For 2 qudits, create proper Bell state
        if num_qudits == 2:
            system.create_generalized_bell_state()
        else:
            # For more qudits, create GHZ-like state
            for i in range(num_qudits):
                system.apply_generalized_hadamard(i)
            
            # Apply generalized CNOT gates
            for i in range(1, num_qudits):
                system.apply_controlled_increment(0, i)
        
        # Get state vector
        state_vector = system.get_state_vector()
        
        # Calculate metrics
        density_matrix = np.outer(state_vector, state_vector.conj())
        
        if num_qudits == 2:
            # Calculate Schmidt decomposition
            psi_matrix = state_vector.reshape((self.dimension, self.dimension))
            U, S, Vh = np.linalg.svd(psi_matrix)
            
            # Calculate entanglement entropy
            entropy = -np.sum(S**2 * np.log2(S**2 + 1e-12))
            
            # Check if maximally entangled
            expected_S = 1 / np.sqrt(self.dimension)
            is_maximally_entangled = np.allclose(S, expected_S, atol=1e-10)
        else:
            # For multiple qudits, calculate subsystem entropy
            subsystem = [0]  # First qudit
            entropy = system.calculate_entropy(subsystem)
            
            # Use singular values from reshaping
            reshaped = state_vector.reshape((self.dimension, -1))
            U, S, Vh = np.linalg.svd(reshaped)
            is_maximally_entangled = False  # Harder to define for multiple qudits
        
        # Measure
        measurements = system.measure(repetitions=5000)
        
        # Analyze measurement correlations
        correlated_prob = 0.0
        for state, prob in measurements['probabilities'].items():
            # Check if all qudits have same value (for Bell/GHZ states)
            if len(set(state)) == 1:  # All digits are the same
                correlated_prob += prob
        
        results = {
            "num_qudits": num_qudits,
            "dimension": self.dimension,
            "schmidt_coefficients": S.tolist() if num_qudits == 2 else [],
            "entanglement_entropy": float(entropy),
            "maximally_entangled": bool(is_maximally_entangled) if num_qudits == 2 else None,
            "correlated_probability": correlated_prob,
            "expected_correlation": 1.0 if num_qudits == 2 else (1/self.dimension),
            "measurements": measurements,
            "duration": time.time() - start_time
        }
        
        print(f"  Entanglement entropy: {entropy:.6f}")
        print(f"  Correlated probability: {correlated_prob:.4f} (expected: {1/self.dimension:.4f})")
        if num_qudits == 2:
            print(f"  Maximally entangled: {'Yes' if is_maximally_entangled else 'No'}")
            print(f"  Schmidt coefficients: {S[:3].tolist()}..." if len(S) > 3 else f"  Schmidt coefficients: {S.tolist()}")
        
        return results
    
    def demo_superposition_states(self, num_qudits: int = 1) -> Dict[str, Any]:
        """Demonstrate superposition in qudits"""
        print(f"\n2. Qudit Superposition States (d={self.dimension})")
        print("-" * 40)
        
        start_time = time.time()
        
        # Create system
        system = QuditSystem(num_qudits, self.dimension, validation_level="warn")
        
        # Create equal superposition across all levels
        superposition_state = np.ones(self.dimension ** num_qudits, dtype=np.complex128)
        superposition_state /= np.linalg.norm(superposition_state)
        system._state = superposition_state
        
        # Measure to verify uniform distribution
        measurements = system.measure(repetitions=10000)
        
        # Calculate uniformity
        probs = list(measurements['probabilities'].values())
        uniform_prob = 1.0 / len(probs)
        
        # Calculate chi-squared test for uniformity
        from scipy import stats
        expected = [uniform_prob] * len(probs)
        chi2, p_value = stats.chisquare(probs, expected)
        
        # Calculate coherence
        density_matrix = np.outer(system._state, system._state.conj())
        coherence = QuantumMetricCalculator.calculate_coherence(density_matrix)
        
        results = {
            "num_qudits": num_qudits,
            "dimension": self.dimension,
            "uniformity_p_value": float(p_value),
            "chi_squared": float(chi2),
            "is_uniform": p_value > 0.05,
            "coherence": float(coherence),
            "expected_coherence": (self.dimension ** num_qudits) - 1,  # Max coherence
            "measurements": measurements,
            "duration": time.time() - start_time
        }
        
        print(f"  Uniformity test p-value: {p_value:.4f} {'(uniform)' if p_value > 0.05 else '(not uniform)'}")
        print(f"  Quantum coherence: {coherence:.2f} (max possible: {(self.dimension ** num_qudits) - 1:.2f})")
        print(f"  Number of basis states: {self.dimension ** num_qudits}")
        
        return results
    
    def demo_quantum_fourier_transform_qudit(self, num_qudits: int = 2) -> Dict[str, Any]:
        """Demonstrate Quantum Fourier Transform on qudits"""
        print(f"\n3. Quantum Fourier Transform (d={self.dimension}, n={num_qudits})")
        print("-" * 40)
        
        start_time = time.time()
        
        # Create system
        system = QuditSystem(num_qudits, self.dimension, validation_level="warn")
        
        # Create a non-uniform initial state
        # For example, |1⟩ for the first qudit, |0⟩ for others
        initial_state = np.zeros(self.dimension ** num_qudits, dtype=np.complex128)
        
        if num_qudits >= 1:
            # Set to |1⟩⊗|0⟩⊗...⊗|0⟩
            index = 1  # Binary 001 in base d
            initial_state[index] = 1.0
        else:
            # Single qudit: |1⟩
            initial_state[1] = 1.0
        
        system._state = initial_state
        
        # Apply QFT to each qudit
        for i in range(num_qudits):
            system.apply_generalized_hadamard(i)
        
        # Get QFT state
        qft_state = system.get_state_vector()
        
        # Apply inverse QFT
        for i in range(num_qudits):
            # Inverse QFT is conjugate transpose of QFT
            omega = np.exp(2j * np.pi / self.dimension)
            matrix = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
            for j in range(self.dimension):
                for k in range(self.dimension):
                    matrix[j, k] = np.conj(omega ** (j * k)) / np.sqrt(self.dimension)
            
            gate = QuantumGate(matrix, f"Inverse_QFT_d{self.dimension}")
            system.apply_gate(gate, [i])
        
        # Should return to original state
        final_state = system.get_state_vector()
        
        # Calculate fidelity
        fidelity = abs(np.vdot(initial_state, final_state))**2
        
        # Measure QFT state to see periodicity
        measurements = system.measure(repetitions=5000)
        
        # Calculate expected probability distribution for QFT of |1⟩
        expected_probs = {}
        d = self.dimension
        for k in range(d ** num_qudits):
            # For QFT of |1⟩, probability should be uniform
            expected_probs[system._index_to_string(k)] = 1.0 / (d ** num_qudits)
        
        results = {
            "num_qudits": num_qudits,
            "dimension": self.dimension,
            "fidelity": float(fidelity),
            "expected_fidelity": 1.0,
            "qft_success": fidelity > 0.99,
            "measurements": measurements,
            "duration": time.time() - start_time
        }
        
        print(f"  QFT fidelity: {fidelity:.6f} (expected: 1.000000)")
        print(f"  QFT success: {'✓' if fidelity > 0.99 else '✗'}")
        print(f"  Hilbert space dimension: {d ** num_qudits}")
        
        return results
    
    def demo_qudit_teleportation(self) -> Dict[str, Any]:
        """Demonstrate teleportation with qudits"""
        print(f"\n4. Qudit Quantum Teleportation (d={self.dimension})")
        print("-" * 40)
        
        start_time = time.time()
        
        # Need at least 3 qudits for teleportation
        system = QuditSystem(3, self.dimension, validation_level="warn")
        
        # Create arbitrary state on qudit 0: ∑_{k=0}^{d-1} α_k|k⟩
        d = self.dimension
        
        # Generate random complex amplitudes
        np.random.seed(42)  # For reproducibility
        alpha = np.random.randn(d) + 1j * np.random.randn(d)
        alpha = alpha / np.linalg.norm(alpha)
        
        # Prepare initial state |ψ⟩⊗|0⟩⊗|0⟩
        initial_state = np.zeros(d**3, dtype=np.complex128)
        for k in range(d):
            # |k⟩⊗|0⟩⊗|0⟩ has index k * d^2
            initial_state[k * d**2] = alpha[k]
        
        system._state = initial_state
        
        # Create generalized Bell pair between qudits 1 and 2
        # Apply generalized Hadamard to qudit 1
        system.apply_generalized_hadamard(1)
        
        # Apply CINC between qudits 1 and 2
        system.apply_controlled_increment(1, 2)
        
        # Teleportation protocol
        # Apply CINC between qudits 0 and 1
        system.apply_controlled_increment(0, 1)
        
        # Apply generalized QFT† (inverse QFT) to qudit 0
        omega = np.exp(2j * np.pi / d)
        qft_inv_matrix = np.zeros((d, d), dtype=np.complex128)
        for j in range(d):
            for k in range(d):
                qft_inv_matrix[j, k] = np.conj(omega ** (j * k)) / np.sqrt(d)
        
        qft_inv_gate = QuantumGate(qft_inv_matrix, "QFT_inverse")
        system.apply_gate(qft_inv_gate, [0])
        
        # Measure qudits 0 and 1
        # In simulation, we calculate the expected state on qudit 2
        # rather than collapsing
        
        # Calculate reduced density matrix of qudit 2
        state_vector = system.get_state_vector()
        
        # Reshape to separate qudit 2
        psi_tensor = state_vector.reshape((d, d, d))
        
        # Trace out qudits 0 and 1
        rho_2 = np.zeros((d, d), dtype=np.complex128)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        rho_2[k, l] += psi_tensor[i, j, k] * np.conj(psi_tensor[i, j, l])
        
        # Theoretically, rho_2 should be |ψ⟩⟨ψ|
        rho_target = np.outer(alpha, np.conj(alpha))
        
        # Calculate fidelity
        fidelity = np.abs(np.trace(rho_2 @ rho_target))
        
        # For pure states, simpler fidelity calculation
        # Extract state of qudit 2
        state_2 = np.zeros(d, dtype=np.complex128)
        for k in range(d):
            # Sum over all states where qudit 2 = k
            for i in range(d):
                for j in range(d):
                    idx = i * d**2 + j * d + k
                    state_2[k] += state_vector[idx]
        
        # Normalize
        norm = np.linalg.norm(state_2)
        if norm > 0:
            state_2 /= norm
        
        simple_fidelity = abs(np.vdot(state_2, alpha))**2
        
        results = {
            "dimension": d,
            "initial_state": alpha.tolist(),
            "teleported_state": state_2.tolist(),
            "fidelity_density_matrix": float(fidelity),
            "fidelity_state_vector": float(simple_fidelity),
            "teleportation_success": simple_fidelity > 0.99,
            "duration": time.time() - start_time
        }
        
        print(f"  Teleportation fidelity: {simple_fidelity:.6f} (expected: 1.000000)")
        print(f"  Success: {'✓' if simple_fidelity > 0.99 else '✗'}")
        print(f"  Dimension: d={d}")
        
        return results
    
    def demo_qudit_entanglement_witness(self) -> Dict[str, Any]:
        """Demonstrate entanglement detection in qudit systems"""
        print(f"\n5. Qudit Entanglement Witness (d={self.dimension})")
        print("-" * 40)
        
        start_time = time.time()
        
        # Create two different 2-qudit states
        d = self.dimension
        
        # State 1: Maximally entangled
        system1 = QuditSystem(2, d, validation_level="warn")
        system1.create_generalized_bell_state()
        
        # State 2: Separable state |0⟩⊗|0⟩
        system2 = QuditSystem(2, d, validation_level="warn")
        
        # Calculate entanglement metrics
        state1 = system1.get_state_vector()
        state2 = system2.get_state_vector()
        
        # Reshape to matrices for Schmidt decomposition
        psi1 = state1.reshape((d, d))
        psi2 = state2.reshape((d, d))
        
        # Singular values
        S1 = np.linalg.svd(psi1, compute_uv=False)
        S2 = np.linalg.svd(psi2, compute_uv=False)
        
        # Entanglement entropies
        E1 = -np.sum(S1**2 * np.log2(S1**2 + 1e-12))
        E2 = -np.sum(S2**2 * np.log2(S2**2 + 1e-12))
        
        # Concurrence (for d=2, but generalized for d>2)
        # Using I-concurrence: C = sqrt(2(1 - Tr(ρ_A²)))
        rho1_A = psi1 @ psi1.conj().T
        rho2_A = psi2 @ psi2.conj().T
        
        concurrence1 = np.sqrt(2 * (1 - np.trace(rho1_A @ rho1_A)))
        concurrence2 = np.sqrt(2 * (1 - np.trace(rho2_A @ rho2_A)))
        
        # Positive Partial Transpose (PPT) criterion
        # For d=2, PPT is necessary and sufficient for separability
        # For d>2, it's only necessary
        
        # Calculate partial transpose of density matrix
        rho1 = np.outer(state1, state1.conj())
        rho2 = np.outer(state2, state2.conj())
        
        # Reshape to perform partial transpose
        rho1_reshaped = rho1.reshape((d, d, d, d))
        rho2_reshaped = rho2.reshape((d, d, d, d))
        
        # Partial transpose on second subsystem
        rho1_pt = np.zeros_like(rho1_reshaped)
        rho2_pt = np.zeros_like(rho2_reshaped)
        
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        rho1_pt[i, j, k, l] = rho1_reshaped[i, l, k, j]
                        rho2_pt[i, j, k, l] = rho2_reshaped[i, l, k, j]
        
        # Reshape back
        rho1_pt = rho1_pt.reshape((d*d, d*d))
        rho2_pt = rho2_pt.reshape((d*d, d*d))
        
        # Check for negative eigenvalues
        eigvals1 = np.linalg.eigvalsh(rho1_pt)
        eigvals2 = np.linalg.eigvalsh(rho2_pt)
        
        has_negative1 = np.any(eigvals1 < -1e-10)
        has_negative2 = np.any(eigvals2 < -1e-10)
        
        results = {
            "dimension": d,
            "state1_entropy": float(E1),
            "state2_entropy": float(E2),
            "state1_concurrence": float(concurrence1),
            "state2_concurrence": float(concurrence2),
            "state1_ppt_negative": bool(has_negative1),
            "state2_ppt_negative": bool(has_negative2),
            "state1_entangled": bool(has_negative1 or E1 > 1e-10),
            "state2_entangled": bool(has_negative2 or E2 > 1e-10),
            "duration": time.time() - start_time
        }
        
        print(f"  State 1 (Bell): Entropy={E1:.4f}, Concurrence={concurrence1:.4f}, Entangled={'Yes' if results['state1_entangled'] else 'No'}")
        print(f"  State 2 (|00⟩): Entropy={E2:.4f}, Concurrence={concurrence2:.4f}, Entangled={'Yes' if results['state2_entangled'] else 'No'}")
        print(f"  PPT criterion: Bell state {'has' if has_negative1 else 'no'} negative eigenvalues")
        
        return results
