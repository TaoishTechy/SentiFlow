"""
True Quantum Metrics Calculator
Scientifically valid quantum metric calculations
"""

import numpy as np
from typing import Tuple, Dict, Any

class QuantumMetricCalculator:
    """Calculate scientifically valid quantum metrics"""
    
    @staticmethod
    def calculate_von_neumann_entropy(density_matrix: np.ndarray) -> float:
        """
        Calculate von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ)
        """
        # Ensure Hermitian
        if not np.allclose(density_matrix, density_matrix.conj().T):
            density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        
        # Remove numerical noise below threshold
        threshold = 1e-14
        eigenvalues = eigenvalues[eigenvalues > threshold]
        
        if len(eigenvalues) == 0:
            return 0.0
        
        # Calculate entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        return float(entropy)
    
    @staticmethod
    def calculate_coherence(density_matrix: np.ndarray, 
                           basis: str = "computational") -> float:
        """
        Calculate quantum coherence using l1-norm of off-diagonal elements
        C(ρ) = Σ_{i≠j} |ρ_ij|
        """
        if basis != "computational":
            # For now, only computational basis supported
            # In production, implement basis transformations
            pass
        
        # Extract off-diagonal elements
        n = density_matrix.shape[0]
        off_diag_sum = 0.0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag_sum += np.abs(density_matrix[i, j])
        
        return off_diag_sum
    
    @staticmethod
    def calculate_purity(density_matrix: np.ndarray) -> float:
        """Calculate purity γ = Tr(ρ²)"""
        return float(np.real(np.trace(density_matrix @ density_matrix)))
    
    @staticmethod
    def calculate_fidelity(state_a: np.ndarray, 
                          state_b: np.ndarray) -> float:
        """Calculate quantum fidelity F = |⟨ψ|φ⟩|²"""
        overlap = np.abs(np.vdot(state_a, state_b))**2
        return float(overlap)
    
    @staticmethod
    def calculate_concurrence(state_vector: np.ndarray, 
                             subsystem_split: Tuple[int, int]) -> float:
        """
        Calculate concurrence for bipartite entanglement
        """
        # Reshape to bipartite structure
        dim_a, dim_b = subsystem_split
        psi_matrix = state_vector.reshape((dim_a, dim_b))
        
        # Calculate concurrence
        if dim_a == 2 and dim_b == 2:  # Qubit case
            # Use Wootters formula
            rho = psi_matrix @ psi_matrix.conj().T
            sigma_y = np.array([[0, -1j], [1j, 0]])
            rho_tilde = np.kron(sigma_y, sigma_y) @ rho.conj() @ np.kron(sigma_y, sigma_y)
            eigenvalues = np.linalg.eigvalsh(rho @ rho_tilde)
            eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
            eigenvalues.sort()
            concurrence = max(0, eigenvalues[-1] - np.sum(eigenvalues[:-1]))
        else:  # Qudit case - generalized concurrence
            # Use I-concurrence
            singular_values = np.linalg.svd(psi_matrix, compute_uv=False)
            concurrence = np.sqrt(2 * (1 - np.sum(singular_values**4)))
        
        return concurrence
    
    @staticmethod
    def calculate_entanglement_entropy(state_vector: np.ndarray, 
                                      partition: int) -> float:
        """
        Calculate entanglement entropy for bipartition
        """
        # Reshape to bipartite
        dim_a = 2 ** partition
        dim_b = len(state_vector) // dim_a
        
        psi_matrix = state_vector.reshape((dim_a, dim_b))
        
        # Calculate reduced density matrix
        rho_a = psi_matrix @ psi_matrix.conj().T
        
        # Calculate entropy
        return QuantumMetricCalculator.calculate_von_neumann_entropy(rho_a)