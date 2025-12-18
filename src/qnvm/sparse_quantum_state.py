import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import sparse
import heapq

class SparseQuantumState:
    """Sparse representation of quantum state"""
    def __init__(self, num_qubits: int, sparsity_threshold: float = 1e-6):
        self.num_qubits = num_qubits
        self.sparsity_threshold = sparsity_threshold
        self.amplitudes = {}  # basis_state -> complex amplitude
        self.norm = 0.0
        self.num_nonzero = 0
    
    def from_dense(self, dense_state: np.ndarray) -> 'SparseQuantumState':
        """Convert dense state to sparse representation"""
        self.amplitudes.clear()
        
        for i, amp in enumerate(dense_state):
            if np.abs(amp) > self.sparsity_threshold:
                self.amplitudes[i] = amp
                self.norm += np.abs(amp) ** 2
        
        self.num_nonzero = len(self.amplitudes)
        return self
    
    def to_dense(self) -> np.ndarray:
        """Convert sparse state to dense representation"""
        dense = np.zeros(2 ** self.num_qubits, dtype=np.complex128)
        for state, amp in self.amplitudes.items():
            dense[state] = amp
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(dense) ** 2))
        if norm > 0:
            dense /= norm
        
        return dense
    
    def apply_gate(self, gate_matrix: np.ndarray, target_qubit: int):
        """Apply gate to sparse state"""
        if gate_matrix.shape != (2, 2):
            raise ValueError("Gate matrix must be 2x2 for single-qubit gate")
        
        new_amplitudes = {}
        
        for basis_state, amp in self.amplitudes.items():
            # Extract target qubit value
            target_bit = (basis_state >> target_qubit) & 1
            
            # Apply gate
            new_amp_0 = gate_matrix[0, target_bit] * amp
            new_amp_1 = gate_matrix[1, target_bit] * amp
            
            # Update basis states
            state_0 = basis_state & ~(1 << target_qubit)  # Set target to 0
            if np.abs(new_amp_0) > self.sparsity_threshold:
                if state_0 in new_amplitudes:
                    new_amplitudes[state_0] += new_amp_0
                else:
                    new_amplitudes[state_0] = new_amp_0
            
            state_1 = basis_state | (1 << target_qubit)  # Set target to 1
            if np.abs(new_amp_1) > self.sparsity_threshold:
                if state_1 in new_amplitudes:
                    new_amplitudes[state_1] += new_amp_1
                else:
                    new_amplitudes[state_1] = new_amp_1
        
        self.amplitudes = new_amplitudes
        self.num_nonzero = len(self.amplitudes)
        
        # Update norm
        self.norm = sum(np.abs(amp) ** 2 for amp in self.amplitudes.values())
    
    def get_entanglement_entropy(self, partition: List[int]) -> float:
        """Calculate entanglement entropy for bipartition"""
        # Simplified calculation
        if not self.amplitudes:
            return 0.0
        
        # Build reduced density matrix
        partition_size = len(partition)
        other_qubits = [i for i in range(self.num_qubits) if i not in partition]
        
        # This is simplified - real implementation would trace out other qubits
        return 0.5  # Placeholder
    
    def compress(self, max_elements: int) -> 'SparseQuantumState':
        """Compress state by keeping only largest amplitudes"""
        if len(self.amplitudes) <= max_elements:
            return self
        
        # Keep largest amplitudes by magnitude
        largest = heapq.nlargest(
            max_elements, 
            self.amplitudes.items(), 
            key=lambda x: np.abs(x[1])
        )
        
        self.amplitudes = dict(largest)
        self.num_nonzero = len(self.amplitudes)
        
        # Renormalize
        self.norm = sum(np.abs(amp) ** 2 for amp in self.amplitudes.values())
        scale = 1.0 / np.sqrt(self.norm)
        for state in self.amplitudes:
            self.amplitudes[state] *= scale
        
        self.norm = 1.0
        return self
    
    def get_stats(self) -> Dict:
        """Get statistics about sparse state"""
        amps = list(self.amplitudes.values())
        magnitudes = [np.abs(amp) for amp in amps]
        
        return {
            'num_qubits': self.num_qubits,
            'num_nonzero': self.num_nonzero,
            'sparsity': 1.0 - (self.num_nonzero / (2 ** self.num_qubits)),
            'norm': self.norm,
            'avg_magnitude': np.mean(magnitudes) if magnitudes else 0,
            'max_magnitude': np.max(magnitudes) if magnitudes else 0,
            'min_magnitude': np.min(magnitudes) if magnitudes else 0
        }

class CompressedState:
    """Compressed quantum state using various techniques"""
    def __init__(self, compression_method: str = 'auto'):
        self.compression_method = compression_method
        self.data = None
        self.metadata = {}
    
    def compress(self, state: np.ndarray, target_ratio: float = 0.1) -> np.ndarray:
        """Compress quantum state to target ratio"""
        if target_ratio >= 1.0:
            return state
        
        if self.compression_method == 'top_k':
            return self._top_k_compression(state, target_ratio)
        elif self.compression_method == 'threshold':
            return self._threshold_compression(state, target_ratio)
        elif self.compression_method == 'randomized':
            return self._randomized_compression(state, target_ratio)
        else:
            return self._auto_compression(state, target_ratio)
    
    def _top_k_compression(self, state: np.ndarray, ratio: float) -> np.ndarray:
        """Keep only largest k amplitudes"""
        k = int(len(state) * ratio)
        if k < 1:
            k = 1
        
        # Find indices of largest amplitudes
        magnitudes = np.abs(state)
        indices = np.argpartition(magnitudes, -k)[-k:]
        
        compressed = np.zeros_like(state)
        compressed[indices] = state[indices]
        
        # Renormalize
        norm = np.linalg.norm(compressed)
        if norm > 0:
            compressed /= norm
        
        self.metadata.update({
            'method': 'top_k',
            'k': k,
            'compression_ratio': ratio,
            'preserved_norm': norm
        })
        
        return compressed
    
    def _threshold_compression(self, state: np.ndarray, ratio: float) -> np.ndarray:
        """Keep amplitudes above threshold"""
        magnitudes = np.abs(state)
        threshold = np.percentile(magnitudes, 100 * (1 - ratio))
        
        compressed = state.copy()
        compressed[magnitudes < threshold] = 0
        
        # Renormalize
        norm = np.linalg.norm(compressed)
        if norm > 0:
            compressed /= norm
        
        self.metadata.update({
            'method': 'threshold',
            'threshold': threshold,
            'nonzero_elements': np.sum(magnitudes >= threshold),
            'compression_ratio': ratio
        })
        
        return compressed
    
    def _auto_compression(self, state: np.ndarray, ratio: float) -> np.ndarray:
        """Automatically choose best compression method"""
        # Try different methods and choose best
        methods = ['top_k', 'threshold']
        best_fidelity = 0
        best_result = state
        
        for method in methods:
            self.compression_method = method
            compressed = self.compress(state, ratio)
            
            # Calculate fidelity with original
            fidelity = np.abs(np.vdot(state, compressed)) ** 2
            
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_result = compressed
        
        self.metadata['best_method'] = self.compression_method
        self.metadata['best_fidelity'] = best_fidelity
        
        return best_result