import numpy as np
from typing import Dict, List, Optional, Tuple
import psutil
import os

class QuantumMemoryManager:
    """Manages quantum state memory with compression and sparsity"""
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.memory_usage = 0.0
        self.states = {}  # id -> state information
        self.compression_stats = {}
    
    def allocate_state(self, num_qubits: int, state_id: str, 
                      sparse_threshold: float = 1e-6) -> Dict:
        """Allocate memory for quantum state with appropriate representation"""
        full_size = 2 ** num_qubits
        dense_memory_gb = full_size * 16 / (1024**3)  # complex128
        
        if dense_memory_gb > self.max_memory_gb:
            # Use sparse/tensor representation
            representation = self._choose_representation(num_qubits, sparse_threshold)
            memory_estimate = self._estimate_memory(num_qubits, representation)
            
            if memory_estimate > self.max_memory_gb:
                raise MemoryError(
                    f"Cannot allocate {memory_estimate:.2f} GB for {num_qubits} qubits. "
                    f"Max: {self.max_memory_gb} GB"
                )
            
            self.states[state_id] = {
                'num_qubits': num_qubits,
                'representation': representation,
                'memory_allocated': memory_estimate,
                'sparse_threshold': sparse_threshold,
                'compression_ratio': memory_estimate / dense_memory_gb
            }
            
            self.memory_usage += memory_estimate
            self.compression_stats[state_id] = {
                'dense_size_gb': dense_memory_gb,
                'actual_size_gb': memory_estimate,
                'compression_ratio': memory_estimate / dense_memory_gb
            }
            
            return self.states[state_id]
        else:
            # Use dense representation
            self.states[state_id] = {
                'num_qubits': num_qubits,
                'representation': 'dense',
                'memory_allocated': dense_memory_gb,
                'compression_ratio': 1.0
            }
            self.memory_usage += dense_memory_gb
            return self.states[state_id]
    
    def _choose_representation(self, num_qubits: int, 
                             sparse_threshold: float) -> str:
        """Choose optimal state representation"""
        if num_qubits <= 16:
            return 'dense'
        elif num_qubits <= 24:
            return 'sparse'
        elif num_qubits <= 28:
            return 'tensor_mps'
        else:  # 29-32 qubits
            return 'tensor_tree'
    
    def _estimate_memory(self, num_qubits: int, representation: str) -> float:
        """Estimate memory usage for different representations"""
        full_size = 2 ** num_qubits
        
        if representation == 'dense':
            return full_size * 16 / (1024**3)  # complex128
        
        elif representation == 'sparse':
            # Assume 1% sparsity for typical circuits
            sparse_elements = full_size * 0.01
            return sparse_elements * 24 / (1024**3)  # index + complex
        
        elif representation == 'tensor_mps':
            # MPS with bond dimension 4
            bond_dim = 4
            memory = 0
            for i in range(num_qubits):
                if i == 0 or i == num_qubits - 1:
                    memory += bond_dim * 2 * 16  # (bond_dim, 2)
                else:
                    memory += bond_dim * 2 * bond_dim * 16  # (bond_dim, 2, bond_dim)
            return memory / (1024**3)
        
        elif representation == 'tensor_tree':
            # Tree tensor network
            return (num_qubits * 64 * 16) / (1024**3)  # Simplified
        
        else:
            raise ValueError(f"Unknown representation: {representation}")
    
    def compress_state(self, state_vector: np.ndarray, 
                      target_qubits: int,
                      compression_ratio: float = 0.1) -> np.ndarray:
        """Compress quantum state with specified ratio"""
        if compression_ratio >= 1.0:
            return state_vector
        
        # Simple compression: keep largest amplitudes
        flat_state = np.abs(state_vector).flatten()
        threshold = np.percentile(flat_state, 100 * (1 - compression_ratio))
        
        mask = np.abs(state_vector) > threshold
        compressed = state_vector * mask
        
        # Renormalize
        norm = np.linalg.norm(compressed)
        if norm > 0:
            compressed /= norm
        
        return compressed
    
    def get_memory_stats(self) -> Dict:
        """Get detailed memory statistics"""
        process = psutil.Process(os.getpid())
        system_memory = psutil.virtual_memory()
        
        return {
            'quantum_memory_usage_gb': self.memory_usage,
            'process_memory_gb': process.memory_info().rss / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_total_gb': system_memory.total / (1024**3),
            'states_allocated': len(self.states),
            'compression_stats': self.compression_stats
        }
    
    def free_state(self, state_id: str):
        """Free memory allocated for state"""
        if state_id in self.states:
            self.memory_usage -= self.states[state_id]['memory_allocated']
            del self.states[state_id]
            if state_id in self.compression_stats:
                del self.compression_stats[state_id]
    
    def clear_all(self):
        """Clear all allocated states"""
        self.states.clear()
        self.compression_stats.clear()
        self.memory_usage = 0.0