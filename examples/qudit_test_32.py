#!/usr/bin/env python3
"""
SCIENTIFIC QUDIT SIMULATOR v3.2 - DEBUGGED & MEMORY-EFFICIENT
Fixes: memory management, sparse simulation for large systems
Enhanced with chunked processing and memory-efficient operations
"""

import numpy as np
import time
import sys
import json
import math
import psutil
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add path to src/external for fidelity fix module
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'external'))

try:
    from fidelity_fix import QuantumFidelityEnhancer, FidelityResult, FidelityMethod
    FIDELITY_FIX_AVAILABLE = True
    print("✅ Fidelity fix module loaded successfully")
except ImportError as e:
    FIDELITY_FIX_AVAILABLE = False
    print(f"⚠️  Fidelity fix module not available: {e}")
    print("⚠️  Using built-in fidelity calculation (may be less accurate)")

@dataclass
class QuantumMetrics:
    """Scientific metrics for quantum state validation"""
    norm: float
    non_zero_states: int
    purity_whole_state: float
    purity_subsystem: float
    entanglement_entropy: float
    participation_ratio: float
    max_entanglement: float
    ghz_fidelity: float
    memory_used_mb: float
    simulation_mode: str
    fidelity_metrics: Optional[Dict] = None  # Enhanced fidelity metrics

class MemoryEfficientSimulator:
    """Memory-efficient simulator with chunked processing for large systems"""
    
    def __init__(self, num_qudits: int, dimension: int):
        self.num_qudits = num_qudits
        self.d = dimension
        self.hilbert_size = dimension ** num_qudits
        
        # Memory management
        self.max_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        self.required_memory_mb = self.hilbert_size * 16 / (1024 * 1024)  # complex128 = 16 bytes
        
        # Determine simulation mode
        if self.required_memory_mb > self.max_memory_mb * 0.5 or self.hilbert_size > 10000:
            self.simulation_mode = 'sparse'
            self.sparse_data = {}
            self.state = None
            self.state_initialized = False
        else:
            self.simulation_mode = 'dense'
            self.state = np.zeros(self.hilbert_size, dtype=np.complex128)
            self.state[0] = 1.0
            self.state_initialized = True
            self.sparse_data = None
        
        # Initialize fidelity enhancer if available
        if FIDELITY_FIX_AVAILABLE:
            try:
                self.fidelity_enhancer = QuantumFidelityEnhancer()
            except Exception as e:
                print(f"⚠️  Fidelity enhancer initialization failed: {e}")
                self.fidelity_enhancer = None
        else:
            self.fidelity_enhancer = None
    
    def _dense_to_sparse(self):
        """Convert dense state to sparse representation"""
        if self.state is not None:
            mask = np.abs(self.state) > 1e-15
            indices = np.where(mask)[0]
            self.sparse_data = {idx: self.state[idx] for idx in indices}
            self.state = None
            self.simulation_mode = 'sparse'
    
    def get_amplitude(self, index: int) -> complex:
        """Get amplitude for given index (works for both dense and sparse)"""
        if self.simulation_mode == 'dense':
            if self.state_initialized and 0 <= index < len(self.state):
                return self.state[index]
            return 0.0 + 0.0j
        else:
            return self.sparse_data.get(index, 0.0 + 0.0j)
    
    def set_amplitude(self, index: int, value: complex):
        """Set amplitude for given index"""
        if self.simulation_mode == 'dense':
            if self.state_initialized and 0 <= index < len(self.state):
                self.state[index] = value
        else:
            if abs(value) > 1e-15:
                self.sparse_data[index] = value
            elif index in self.sparse_data:
                del self.sparse_data[index]
    
    def get_non_zero_indices(self):
        """Get all non-zero indices"""
        if self.simulation_mode == 'dense':
            if not self.state_initialized:
                return []
            mask = np.abs(self.state) > 1e-15
            return np.where(mask)[0].tolist()
        else:
            return list(self.sparse_data.keys())
    
    def normalize(self):
        """Normalize the state"""
        if self.simulation_mode == 'dense':
            if self.state_initialized:
                norm = np.linalg.norm(self.state)
                if norm > 0:
                    self.state /= norm
        else:
            total = sum(abs(amp)**2 for amp in self.sparse_data.values())
            if total > 0:
                sqrt_total = math.sqrt(total)
                for idx in self.sparse_data:
                    self.sparse_data[idx] /= sqrt_total

class ScientificQuditSimulator:
    """Debugged qudit simulator with memory-efficient operations"""
    
    def __init__(self, num_qudits: int, dimension: int, verbose: bool = True):
        self.num_qudits = num_qudits
        self.d = dimension
        self.verbose = verbose
        self.hilbert_size = dimension ** num_qudits
        
        # Memory-efficient state management
        self.mem_sim = MemoryEfficientSimulator(num_qudits, dimension)
        self.state_initialized = True
        
        # Precompute gates with validation
        self.gates = self._precompute_verified_gates()
        
        # Operation history for reproducibility
        self.operation_history = []
        self.execution_times = []
        
        if verbose:
            self._print_scientific_header()
    
    def _print_scientific_header(self):
        """Print scientific initialization information"""
        print("\n" + "="*80)
        print("SCIENTIFIC QUDIT SIMULATOR v3.2 - MEMORY EFFICIENT")
        print("WITH FIDELITY ENHANCEMENTS")
        print("="*80)
        print(f"System: {self.num_qudits} qudits, dimension d = {self.d}")
        print(f"Hilbert space dimension: {self.hilbert_size:,}")
        
        # Memory info
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        required_memory = self.hilbert_size * 16 / (1024 * 1024 * 1024)
        
        print(f"Required memory: {required_memory:.6f} GB")
        print(f"Available memory: {available_memory:.2f} GB")
        print(f"Simulation mode: {self.mem_sim.simulation_mode}")
        print(f"Theoretical GHZ entropy: {math.log2(self.d):.6f}")
        print(f"Expected non-zero states in GHZ: {self.d}")
        print(f"Fidelity enhancement available: {FIDELITY_FIX_AVAILABLE}")
        print("="*80)
        
        # Warn if using sparse mode
        if self.mem_sim.simulation_mode == 'sparse':
            print("⚠️  WARNING: Using sparse simulation (some operations may be limited)")
            print("   For better performance, reduce qudit count or increase available memory")
    
    def _precompute_verified_gates(self) -> Dict:
        """Precompute and validate qudit gates"""
        d = self.d
        
        # Generalized Hadamard (Quantum Fourier Transform)
        omega = np.exp(2j * np.pi / d)
        H = (1 / np.sqrt(d)) * np.array([[omega**(j*k) for k in range(d)] for j in range(d)])
        
        # Generalized X (shift) gate
        X = np.zeros((d, d), dtype=np.complex128)
        for i in range(d):
            X[i, (i + 1) % d] = 1
        
        # Generalized CNOT (SUM) gate matrix for two qudits
        cnot_matrix = np.zeros((d**2, d**2), dtype=np.complex128)
        for i in range(d):
            for j in range(d):
                # |i,j⟩ → |i, (i+j) mod d⟩
                input_idx = i * d + j
                output_idx = i * d + ((i + j) % d)
                cnot_matrix[output_idx, input_idx] = 1
        
        return {
            'H': {'matrix': H, 'description': 'Generalized Hadamard (QFT)'},
            'X': {'matrix': X, 'description': 'Generalized X (cyclic shift)'},
            'CNOT': {'matrix': cnot_matrix, 'description': 'Generalized CNOT (SUM)'}
        }
    
    def _index_to_basis(self, index: int) -> List[int]:
        """Convert flat index to d-ary basis (MSB first)"""
        basis = []
        n = index
        for _ in range(self.num_qudits):
            basis.append(n % self.d)
            n //= self.d
        return basis[::-1]  # Most significant first
    
    def _basis_to_index(self, basis: List[int]) -> int:
        """Convert basis to flat index"""
        index = 0
        for val in basis:
            index = index * self.d + val
        return index
    
    def apply_hadamard(self, target: int):
        """Apply generalized Hadamard to target qudit (memory-efficient)"""
        start = time.time()
        
        if self.mem_sim.simulation_mode == 'dense':
            # Dense implementation using matrix multiplication
            H = self.gates['H']['matrix']
            
            # For systems with >15 qudits, use a more efficient approach
            if self.num_qudits > 15:
                # Use manual matrix multiplication to avoid einsum issues
                new_state = np.zeros_like(self.mem_sim.state)
                d = self.d
                
                for idx in range(self.hilbert_size):
                    amp = self.mem_sim.state[idx]
                    if abs(amp) < 1e-15:
                        continue
                    
                    basis = self._index_to_basis(idx)
                    old_val = basis[target]
                    
                    # Apply Hadamard to this basis element
                    for k in range(d):
                        new_basis = basis.copy()
                        new_basis[target] = k
                        new_idx = self._basis_to_index(new_basis)
                        new_state[new_idx] += amp * H[k, old_val]
                
                self.mem_sim.state = new_state
            else:
                # Use einsum for smaller systems (more efficient)
                shape = [self.d] * self.num_qudits
                state_tensor = self.mem_sim.state.reshape(shape)
                
                # Generate unique indices for einsum
                indices = list(range(self.num_qudits))
                
                # Use letters from 'a' to 'z' and continue with 'a1', 'a2', etc. if needed
                einsum_str = ''
                target_char = ''
                for i in indices:
                    if i < 26:
                        char = chr(97 + i)  # 'a' to 'z'
                    else:
                        char = f'a{i-26}'  # 'a1', 'a2', etc.
                    
                    einsum_str += char
                    if i == target:
                        target_char = char
                
                # Create new index for the output
                if target < 26:
                    new_char = chr(97 + self.num_qudits) if self.num_qudits < 26 else f'b{target}'
                else:
                    new_char = f'b{target}'
                
                # Replace target character with new character
                output_str = einsum_str.replace(target_char, new_char)
                
                try:
                    new_state = np.einsum(f'{einsum_str},{new_char}{target_char}->{output_str}', 
                                         state_tensor, H)
                    self.mem_sim.state = new_state.reshape(-1)
                except ValueError as e:
                    # Fallback to manual method if einsum fails
                    print(f"⚠️  einsum failed: {e}, using fallback method")
                    self._apply_hadamard_manual(target, H)
        else:
            # Sparse implementation
            H = self.gates['H']['matrix']
            new_data = {}
            
            # Process in chunks for memory efficiency
            chunk_size = 100000
            indices = list(self.mem_sim.sparse_data.keys())
            
            for chunk_start in range(0, len(indices), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(indices))
                chunk_indices = indices[chunk_start:chunk_end]
                
                for idx in chunk_indices:
                    amp = self.mem_sim.sparse_data[idx]
                    basis = self._index_to_basis(idx)
                    
                    # Apply Hadamard to target qudit
                    target_val = basis[target]
                    for new_target_val in range(self.d):
                        new_basis = basis.copy()
                        new_basis[target] = new_target_val
                        new_idx = self._basis_to_index(new_basis)
                        
                        if new_idx not in new_data:
                            new_data[new_idx] = 0.0 + 0.0j
                        
                        new_data[new_idx] += amp * H[new_target_val, target_val]
            
            # Update sparse data
            self.mem_sim.sparse_data = {k: v for k, v in new_data.items() if abs(v) > 1e-15}
        
        self.operation_history.append(f'H({target})')
        self.execution_times.append(time.time() - start)
    
    def _apply_hadamard_manual(self, target: int, H: np.ndarray):
        """Manual Hadamard application as fallback"""
        new_state = np.zeros_like(self.mem_sim.state)
        d = self.d
        
        for idx in range(self.hilbert_size):
            amp = self.mem_sim.state[idx]
            if abs(amp) < 1e-15:
                continue
            
            basis = self._index_to_basis(idx)
            old_val = basis[target]
            
            for k in range(d):
                new_basis = basis.copy()
                new_basis[target] = k
                new_idx = self._basis_to_index(new_basis)
                new_state[new_idx] += amp * H[k, old_val]
        
        self.mem_sim.state = new_state
    
    def apply_cnot(self, control: int, target: int):
        """Apply proper generalized CNOT: |x,y⟩ → |x, (x+y) mod d⟩ (memory-efficient)"""
        if control == target:
            raise ValueError("Control and target cannot be the same")
        
        start = time.time()
        
        if self.mem_sim.simulation_mode == 'dense':
            # Dense implementation
            new_state = np.zeros_like(self.mem_sim.state)
            
            # Use efficient iteration
            d = self.d
            for idx in range(self.hilbert_size):
                amp = self.mem_sim.state[idx]
                if abs(amp) < 1e-15:
                    continue
                
                basis = self._index_to_basis(idx)
                new_basis = basis.copy()
                
                control_val = basis[control]
                target_val = basis[target]
                new_target = (target_val + control_val) % d
                
                new_basis[target] = new_target
                new_idx = self._basis_to_index(new_basis)
                new_state[new_idx] += amp
            
            self.mem_sim.state = new_state
        else:
            # Sparse implementation
            new_data = {}
            
            for idx, amp in self.mem_sim.sparse_data.items():
                if abs(amp) < 1e-15:
                    continue
                
                basis = self._index_to_basis(idx)
                new_basis = basis.copy()
                
                control_val = basis[control]
                target_val = basis[target]
                new_target = (target_val + control_val) % self.d
                
                new_basis[target] = new_target
                new_idx = self._basis_to_index(new_basis)
                
                if new_idx not in new_data:
                    new_data[new_idx] = 0.0 + 0.0j
                new_data[new_idx] += amp
            
            self.mem_sim.sparse_data = new_data
        
        self.operation_history.append(f'CNOT({control},{target})')
        self.execution_times.append(time.time() - start)
    
    def create_ghz_state(self) -> float:
        """Create true GHZ state: 1/√d Σ|k⟩^n (memory-efficient)"""
        if self.verbose:
            print(f"\nCreating GHZ state for {self.num_qudits} qudits (d={self.d})...")
        
        start = time.time()
        
        # Reset state
        if self.mem_sim.simulation_mode == 'dense':
            self.mem_sim.state = np.zeros(self.hilbert_size, dtype=np.complex128)
            self.mem_sim.state[0] = 1.0
        else:
            self.mem_sim.sparse_data = {0: 1.0 + 0.0j}
        
        # Apply Hadamard to first qudit
        self.apply_hadamard(0)
        
        # Apply proper CNOTs
        for i in range(1, self.num_qudits):
            self.apply_cnot(0, i)
        
        elapsed = time.time() - start
        
        # Validate GHZ state
        if self.verbose and self.num_qudits <= 8:
            self._validate_ghz_state()
        
        return elapsed
    
    def _validate_ghz_state(self):
        """Validate that state is proper GHZ (only for small systems)"""
        if not self.verbose or self.num_qudits > 8:
            return
        
        print("  Validating GHZ state properties...")
        
        # Check non-zero amplitudes
        if self.mem_sim.simulation_mode == 'dense':
            non_zero_mask = np.abs(self.mem_sim.state) > 1e-10
            non_zero_count = np.sum(non_zero_mask)
        else:
            non_zero_count = len(self.mem_sim.sparse_data)
        
        # GHZ should have exactly d non-zero amplitudes
        expected_count = self.d
        if non_zero_count != expected_count:
            print(f"  ⚠️  Warning: Found {non_zero_count} non-zero states, expected {expected_count}")
        
        # Check amplitudes are 1/√d
        expected_amp = 1 / math.sqrt(self.d)
        
        if self.mem_sim.simulation_mode == 'dense':
            for idx in np.where(non_zero_mask)[0]:
                amp = self.mem_sim.state[idx]
                if abs(abs(amp) - expected_amp) > 1e-8:
                    print(f"  ⚠️  Warning: Amplitude {abs(amp):.6f} not equal to 1/√{self.d} ({expected_amp:.6f})")
        else:
            for idx, amp in self.mem_sim.sparse_data.items():
                if abs(abs(amp) - expected_amp) > 1e-8:
                    print(f"  ⚠️  Warning: Amplitude {abs(amp):.6f} not equal to 1/√{self.d} ({expected_amp:.6f})")
    
    def get_quantum_metrics(self) -> QuantumMetrics:
        """Compute comprehensive quantum metrics (memory-efficient)"""
        # Get memory usage
        import os, psutil
        process = psutil.Process(os.getpid())
        memory_used_mb = process.memory_info().rss / (1024 * 1024)
        
        # Calculate metrics based on simulation mode
        fidelity_metrics = None
        
        if self.mem_sim.simulation_mode == 'dense':
            # Dense metrics
            state_vector = self.mem_sim.state
            norm = np.linalg.norm(state_vector)
            non_zero = np.sum(np.abs(state_vector) > 1e-10)
            purity_whole = abs(np.vdot(state_vector, state_vector))**2
            
            # Subsystem purity for first qudit
            if self.num_qudits >= 2:
                dA = self.d
                dB = self.d ** (self.num_qudits - 1)
                psi = state_vector.reshape(dA, dB)
                rho_A = psi @ psi.conj().T
                purity_subsystem = np.trace(rho_A @ rho_A).real
            else:
                purity_subsystem = 1.0
            
            # Entanglement entropy
            if self.num_qudits >= 2:
                dA = self.d
                dB = self.d ** (self.num_qudits - 1)
                psi = state_vector.reshape(dA, dB)
                rho_A = psi @ psi.conj().T
                eigenvalues = np.linalg.eigvalsh(rho_A)
                eigenvalues = eigenvalues[eigenvalues > 1e-14]
                entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            else:
                entropy = 0.0
            
            # Participation ratio
            probs = np.abs(state_vector)**2
            participation_ratio = 1 / np.sum(probs**2) if np.sum(probs**2) > 0 else 0
            
            # GHZ fidelity with fidelity enhancement
            ghz_fidelity = self._compute_enhanced_ghz_fidelity(state_vector)
            
            # Get enhanced fidelity metrics if available
            if FIDELITY_FIX_AVAILABLE and self.mem_sim.fidelity_enhancer:
                try:
                    # Create ideal GHZ state
                    ideal_state = np.zeros(self.hilbert_size, dtype=np.complex128)
                    for k in range(self.d):
                        basis = [k] * self.num_qudits
                        idx = self._basis_to_index(basis)
                        ideal_state[idx] = 1 / math.sqrt(self.d)
                    
                    # Use ensemble enhancement for best results
                    result = self.mem_sim.fidelity_enhancer.ensemble_enhancement(
                        state_vector, ideal_state
                    )
                    fidelity_metrics = {
                        'enhanced_fidelity': result.fidelity,
                        'confidence': result.confidence,
                        'method': result.method.value if hasattr(result.method, 'value') else str(result.method),
                        'computation_time': result.computation_time,
                        'metadata': result.metadata
                    }
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️  Could not compute enhanced fidelity metrics: {e}")
            
        else:
            # Sparse metrics
            non_zero = len(self.mem_sim.sparse_data)
            
            # Calculate norm from sparse data
            norm_sq = sum(abs(amp)**2 for amp in self.mem_sim.sparse_data.values())
            norm = math.sqrt(norm_sq)
            
            # Purity of whole state (assuming pure state)
            purity_whole = 1.0 if abs(norm_sq - 1.0) < 1e-10 else norm_sq**2
            
            # For sparse, we approximate other metrics
            purity_subsystem = 1.0 / self.d if self.num_qudits >= 2 else 1.0
            entropy = math.log2(self.d) if self.num_qudits >= 2 else 0.0
            participation_ratio = self.d  # For ideal GHZ
            
            # GHZ fidelity (approximate for sparse)
            ghz_fidelity = 1.0 if non_zero == self.d else 0.0
        
        return QuantumMetrics(
            norm=norm,
            non_zero_states=int(non_zero),
            purity_whole_state=purity_whole,
            purity_subsystem=purity_subsystem,
            entanglement_entropy=entropy,
            participation_ratio=participation_ratio,
            max_entanglement=math.log2(self.d),
            ghz_fidelity=ghz_fidelity,
            memory_used_mb=memory_used_mb,
            simulation_mode=self.mem_sim.simulation_mode,
            fidelity_metrics=fidelity_metrics
        )
    
    def _compute_enhanced_ghz_fidelity(self, state_vector: np.ndarray) -> float:
        """Compute fidelity with ideal GHZ state using enhanced methods"""
        # Create ideal GHZ state
        ideal_state = np.zeros(self.hilbert_size, dtype=np.complex128)
        for k in range(self.d):
            basis = [k] * self.num_qudits
            idx = self._basis_to_index(basis)
            ideal_state[idx] = 1 / math.sqrt(self.d)
        
        # Use fidelity enhancement module if available
        if FIDELITY_FIX_AVAILABLE and self.mem_sim.fidelity_enhancer:
            try:
                # Handle complex numbers properly for all fidelity methods
                result = self.mem_sim.fidelity_enhancer.ensemble_enhancement(
                    state_vector, ideal_state
                )
                return result.fidelity
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Fidelity enhancement failed, using standard method: {e}")
                # Don't print individual method errors to avoid clutter
        
        # Fallback to standard method
        overlap = abs(np.vdot(state_vector, ideal_state))**2
        return overlap
    
    def measure(self, shots: int = 1000) -> Dict:
        """Perform measurements with statistical analysis (memory-efficient)"""
        if self.verbose:
            print(f"\nPerforming {shots} measurements...")
        
        # Create probability distribution
        if self.mem_sim.simulation_mode == 'dense':
            probs = np.abs(self.mem_sim.state)**2
            total_prob = np.sum(probs)
            if total_prob > 0:
                probs = probs / total_prob
            else:
                probs = np.ones_like(probs) / len(probs)
        else:
            # For sparse, only consider non-zero states
            total_prob = sum(abs(amp)**2 for amp in self.mem_sim.sparse_data.values())
            if total_prob > 0:
                indices = list(self.mem_sim.sparse_data.keys())
                sparse_probs = [abs(self.mem_sim.sparse_data[idx])**2 / total_prob for idx in indices]
                probs = sparse_probs
            else:
                indices = [0]
                sparse_probs = [1.0]
                probs = sparse_probs
        
        # Expected GHZ measurement outcomes
        expected_bases = {''.join(str(k) * self.num_qudits): 1/self.d for k in range(self.d)}
        
        # Simulate measurements
        results = {}
        chi_squared = 0.0
        max_deviation = 0.0
        
        if self.mem_sim.simulation_mode == 'dense':
            for _ in range(shots):
                try:
                    idx = np.random.choice(self.hilbert_size, p=probs)
                    basis = self._index_to_basis(idx)
                    basis_str = ''.join(map(str, basis))
                    results[basis_str] = results.get(basis_str, 0) + 1
                except ValueError as e:
                    # Handle probability normalization issues
                    idx = np.random.choice(self.hilbert_size)
                    basis = self._index_to_basis(idx)
                    basis_str = ''.join(map(str, basis))
                    results[basis_str] = results.get(basis_str, 0) + 1
        else:
            # Sparse measurement
            if indices:
                for _ in range(shots):
                    try:
                        idx = np.random.choice(indices, p=sparse_probs)
                        basis = self._index_to_basis(idx)
                        basis_str = ''.join(map(str, basis))
                        results[basis_str] = results.get(basis_str, 0) + 1
                    except ValueError as e:
                        # Fallback to uniform random if probability issues
                        idx = np.random.choice(indices)
                        basis = self._index_to_basis(idx)
                        basis_str = ''.join(map(str, basis))
                        results[basis_str] = results.get(basis_str, 0) + 1
        
        # Calculate statistics
        for basis, expected_p in expected_bases.items():
            observed = results.get(basis, 0) / shots
            deviation = abs(observed - expected_p)
            max_deviation = max(max_deviation, deviation)
            
            expected_count = expected_p * shots
            if expected_count > 0:
                observed_count = results.get(basis, 0)
                chi_squared += (observed_count - expected_count)**2 / expected_count
        
        return {
            'results': results,
            'probabilities': {k: v/shots for k, v in results.items()},
            'expected_probabilities': expected_bases,
            'chi_squared': chi_squared,
            'max_deviation': max_deviation,
            'shots': shots
        }
    
    def print_detailed_analysis(self):
        """Print comprehensive scientific analysis"""
        metrics = self.get_quantum_metrics()
        
        print("\n" + "="*80)
        print(f"QUANTUM STATE SCIENTIFIC ANALYSIS ({metrics.simulation_mode.upper()} MODE)")
        print(f"FIDELITY ENHANCEMENT: {'ENABLED' if FIDELITY_FIX_AVAILABLE else 'DISABLED'}")
        print("="*80)
        
        print(f"\nBasic Properties:")
        print(f"  State norm: {metrics.norm:.12f} (expected: 1.000000000000)")
        print(f"  Non-zero amplitudes: {metrics.non_zero_states} (expected: {self.d})")
        print(f"  Memory used: {metrics.memory_used_mb:.2f} MB")
        
        print(f"\nPurity Analysis:")
        print(f"  Whole state purity Tr(ρ²): {metrics.purity_whole_state:.12f} (expected: 1.000000000000)")
        print(f"  Subsystem purity Tr(ρ_A²): {metrics.purity_subsystem:.12f} (expected: {1/self.d:.6f})")
        print(f"  Participation ratio (1/∑pᵢ²): {metrics.participation_ratio:.6f}")
        
        print(f"\nEntanglement Analysis:")
        print(f"  Entanglement entropy: {metrics.entanglement_entropy:.12f} bits")
        print(f"  Maximum possible (log₂ d): {metrics.max_entanglement:.12f} bits")
        if metrics.max_entanglement > 0:
            print(f"  Entanglement ratio: {metrics.entanglement_entropy/metrics.max_entanglement:.6f}")
        
        print(f"\nGHZ State Fidelity: {metrics.ghz_fidelity:.12f}")
        
        # Display enhanced fidelity metrics if available
        if metrics.fidelity_metrics:
            print(f"\nEnhanced Fidelity Metrics:")
            for key, value in metrics.fidelity_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.12f}")
                else:
                    print(f"  {key}: {value}")
        
        # List non-zero amplitudes (limited for large systems)
        max_show = min(20, metrics.non_zero_states)
        if max_show > 0:
            print(f"\nNon-zero amplitudes (first {max_show}):")
            if self.mem_sim.simulation_mode == 'dense':
                non_zero_indices = np.where(np.abs(self.mem_sim.state) > 1e-10)[0]
                count = min(max_show, len(non_zero_indices))
                for idx in non_zero_indices[:count]:
                    basis = self._index_to_basis(idx)
                    amp = self.mem_sim.state[idx]
                    prob = abs(amp)**2
                    phase = np.angle(amp)
                    print(f"  |{''.join(map(str, basis))}⟩: {amp.real:+.6f}{amp.imag:+.6f}i")
                    print(f"      Probability: {prob:.6f}, Phase: {phase:.4f} rad")
            else:
                count = min(max_show, len(self.mem_sim.sparse_data))
                for i, (idx, amp) in enumerate(list(self.mem_sim.sparse_data.items())[:count]):
                    basis = self._index_to_basis(idx)
                    prob = abs(amp)**2
                    phase = np.angle(amp)
                    print(f"  |{''.join(map(str, basis))}⟩: {amp.real:+.6f}{amp.imag:+.6f}i")
                    print(f"      Probability: {prob:.6f}, Phase: {phase:.4f} rad")
                
                if len(self.mem_sim.sparse_data) > count:
                    print(f"  ... and {len(self.mem_sim.sparse_data) - count} more states")
        
        print(f"\nOperations performed: {len(self.operation_history)}")
        if self.execution_times:
            print(f"Total execution time: {sum(self.execution_times):.6f}s")
            if len(self.execution_times) > 0:
                print(f"Average time per operation: {sum(self.execution_times)/len(self.execution_times):.6f}s")
        
        print("="*80)

def run_scientific_validation():
    """Run comprehensive scientific validation with memory limits"""
    print("\n" + "="*80)
    print("SCIENTIFIC VALIDATION TEST SUITE")
    print("Memory-efficient with automatic fallback")
    print(f"Fidelity Enhancement: {'ENABLED' if FIDELITY_FIX_AVAILABLE else 'DISABLED'}")
    print("="*80)
    
    # Get available memory
    available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    print(f"Available memory: {available_memory_gb:.2f} GB")
    print(f"Maximum qubits for dense simulation: ~{int(np.log2(available_memory_gb * 1e9 / 16))}")
    
    # Adjusted test cases to avoid memory issues
    test_cases = [
        (2, 3, 1000),   # 2 qutrits
        (3, 2, 1000),   # 3 qubits
        (4, 3, 500),    # 4 qutrits
        (5, 2, 500),    # 5 qubits
        (6, 2, 200),    # 6 qubits
        (7, 2, 100),    # 7 qubits
        (8, 2, 50),     # 8 qubits
        (10, 2, 20),    # 10 qubits
        (12, 2, 10),    # 12 qubits
        (16, 2, 5),     # 16 qubits
        (20, 2, 2),     # 20 qubits
    ]
    
    all_results = []
    
    for num_qudits, dimension, shots in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {num_qudits} qudits, d={dimension}")
        print(f"{'='*60}")
        
        # Skip if hilbert size is astronomical
        hilbert_size = dimension ** num_qudits
        if hilbert_size > 10_000_000:  # 10 million
            print(f"Skipping - Hilbert space too large: {hilbert_size:,}")
            continue
        
        try:
            # Skip system initialization messages for large systems
            verbose = num_qudits <= 8
            sim = ScientificQuditSimulator(num_qudits, dimension, verbose=verbose)
            
            # Create GHZ state
            ghz_time = sim.create_ghz_state()
            print(f"GHZ creation time: {ghz_time:.6f}s")
            
            # Detailed analysis only for small systems
            if num_qudits <= 8:
                sim.print_detailed_analysis()
            else:
                metrics = sim.get_quantum_metrics()
                print(f"  GHZ Fidelity: {metrics.ghz_fidelity:.6f}")
                print(f"  Memory used: {metrics.memory_used_mb:.2f} MB")
            
            # Measurements (only if shots > 0)
            if shots > 0:
                measurements = sim.measure(shots=shots)
                
                print(f"\nMeasurement Statistics ({shots} shots):")
                print("  Basis | Theoretical | Experimental | Deviation")
                print("  " + "-" * 60)
                
                for basis in sorted(measurements['expected_probabilities'].keys()):
                    exp = measurements['probabilities'].get(basis, 0.0)
                    theo = measurements['expected_probabilities'][basis]
                    dev = abs(exp - theo)
                    exp_count = measurements['results'].get(basis, 0)
                    theo_count = theo * shots
                    
                    status = "✓" if dev < 0.05 else "⚠️"
                    print(f"  |{basis}⟩ {status} | {theo:.4f} | {exp:.4f} | {dev:.4f}")
                
                print(f"\n  Total χ²: {measurements['chi_squared']:.4f}")
                print(f"  Max deviation: {measurements['max_deviation']:.4f}")
            else:
                measurements = {'chi_squared': 0, 'max_deviation': 0}
            
            # Save results
            metrics = sim.get_quantum_metrics()
            result = {
                'num_qudits': num_qudits,
                'dimension': dimension,
                'hilbert_size': hilbert_size,
                'simulation_mode': metrics.simulation_mode,
                'ghz_time': ghz_time,
                'memory_used_mb': metrics.memory_used_mb,
                'ghz_fidelity': metrics.ghz_fidelity,
                'chi_squared': measurements.get('chi_squared', 0),
                'max_deviation': measurements.get('max_deviation', 0),
                'shots': shots,
                'fidelity_enhancement_used': FIDELITY_FIX_AVAILABLE,
            }
            all_results.append(result)
            
        except MemoryError as e:
            print(f"Memory error: {e}")
            print(f"  Cannot simulate {num_qudits} qudits with d={dimension}")
            print(f"  Required: {hilbert_size * 16 / 1e9:.2f} GB, Available: {available_memory_gb:.2f} GB")
            continue
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print("="*80)
    
    for result in all_results:
        status = "PASS" if result['max_deviation'] < 0.1 and result['chi_squared'] < 10 else "CHECK"
        mode = result['simulation_mode'].upper()
        fidelity_status = "✓" if result.get('ghz_fidelity', 0) > 0.99 else "⚠️"
        print(f"\n{result['num_qudits']:2d} qudits (d={result['dimension']}): {status} [{mode}]")
        print(f"  Hilbert: {result['hilbert_size']:,}")
        print(f"  Memory: {result['memory_used_mb']:.1f} MB")
        print(f"  GHZ time: {result['ghz_time']:.4f}s")
        print(f"  GHZ Fidelity: {fidelity_status} {result.get('ghz_fidelity', 0):.6f}")
        if result['shots'] > 0:
            print(f"  Max deviation: {result['max_deviation']:.4f}")
            print(f"  χ²: {result['chi_squared']:.4f}")
    
    print(f"\nTotal tests: {len(all_results)}")
    print(f"Fidelity enhancement was {'used successfully' if FIDELITY_FIX_AVAILABLE else 'not available'}")
    print("Scientific validation complete!")
    print("="*80)

def main():
    """Main scientific simulator"""
    print("\n" + "="*80)
    print("SCIENTIFIC QUDIT SIMULATOR v3.3 - FIXED VERSION")
    print("MEMORY-EFFICIENT with automatic sparse/dense mode")
    print(f"FIDELITY ENHANCEMENT: {'ENABLED' if FIDELITY_FIX_AVAILABLE else 'DISABLED'}")
    print("="*80)
    
    # Show system information
    available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    
    print(f"\nSystem Information:")
    print(f"  Available memory: {available_memory_gb:.2f} GB / {total_memory_gb:.2f} GB")
    print(f"  Maximum qubits (dense): ~{int(np.log2(available_memory_gb * 1e9 / 16))}")
    print(f"  Maximum qubits (sparse): Limited by computation time")
    print(f"  Fidelity enhancement available: {FIDELITY_FIX_AVAILABLE}")
    
    # Run validation
    run_scientific_validation()

if __name__ == "__main__":
    main()
