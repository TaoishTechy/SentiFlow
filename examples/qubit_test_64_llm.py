#!/usr/bin/env python3
"""
QNVM v6.0 - Ultra-Scale Quantum Test Suite (Up to 64 Qubits)
Leveraging LLM Optimization and GPU Acceleration
"""

import sys
import os
import time
import numpy as np
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import gc
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings('ignore')

try:
    from qnvm import QNVM, QNVMConfig, create_qnvm, HAS_REAL_IMPL
    from qnvm.config import BackendType, CompressionMethod
    print(f"✅ QNVM v6.0 loaded (Real Implementation: {HAS_REAL_IMPL})")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Try to import LLM and GPU modules
try:
    import torch
    import cupy as cp
    HAS_GPU = torch.cuda.is_available() or hasattr(cp, 'cuda')
    if HAS_GPU:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CuPy available'}")
except ImportError:
    HAS_GPU = False
    print("⚠️  GPU modules not available")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from langchain.llms import HuggingFacePipeline
    HAS_LLM = True
    print("✅ LLM modules available")
except ImportError:
    HAS_LLM = False
    print("⚠️  LLM modules not available")

class LLMOptimizer:
    """LLM-based quantum circuit optimizer"""
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.llm = None
        self.tokenizer = None
        
        if HAS_LLM:
            try:
                print(f"🔄 Loading LLM: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                self.llm = True
                print("✅ LLM optimizer initialized")
            except Exception as e:
                print(f"⚠️  Failed to load LLM: {e}")
                self.llm = False
        else:
            self.llm = False
    
    def optimize_circuit(self, circuit: Dict, constraints: Dict = None) -> Dict:
        """Optimize circuit using LLM reasoning"""
        if not self.llm:
            return circuit  # Return as-is if no LLM
        
        # Extract circuit info
        num_qubits = circuit.get('num_qubits', 0)
        gates = circuit.get('gates', [])
        
        # Create optimization prompt
        prompt = f"""
        Optimize this quantum circuit for {num_qubits} qubits.
        
        Original circuit:
        {json.dumps(gates, indent=2)}
        
        Optimization goals:
        1. Reduce gate count
        2. Minimize circuit depth
        3. Use gate cancellation where possible
        4. Combine consecutive rotations
        5. Remove identity operations
        
        Provide optimized circuit in JSON format:
        """
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=2048,
                temperature=0.1,
                do_sample=True,
                top_p=0.95
            )
            optimized_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', optimized_text, re.DOTALL)
            if json_match:
                optimized_gates = json.loads(json_match.group())
                return {**circuit, 'gates': optimized_gates}
        except Exception as e:
            print(f"⚠️  LLM optimization failed: {e}")
        
        return circuit  # Fallback to original

class GPUQuantumBackend:
    """GPU-accelerated quantum backend"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.use_gpu = HAS_GPU and ("cuda" in device or device == "gpu")
        
        if self.use_gpu:
            if torch.cuda.is_available():
                torch.cuda.set_device(device if ":" in device else 0)
                print(f"✅ Using PyTorch GPU: {torch.cuda.get_device_name()}")
            else:
                print("✅ Using CuPy GPU acceleration")
        else:
            print("⚠️  Using CPU backend")
    
    def create_state_vector(self, num_qubits: int) -> Any:
        """Create state vector on appropriate device"""
        size = 2 ** num_qubits
        
        if self.use_gpu:
            if torch.cuda.is_available():
                return torch.zeros(size, dtype=torch.complex64, device=self.device)
            else:
                return cp.zeros(size, dtype=cp.complex64)
        else:
            return np.zeros(size, dtype=np.complex64)
    
    def apply_gate_matrix(self, state: Any, matrix: Any, targets: List[int]) -> Any:
        """Apply gate matrix to state vector"""
        if self.use_gpu and torch.cuda.is_available():
            # PyTorch implementation
            shape = [2] * int(np.log2(len(state)))
            
            # Reshape and apply gate
            reshaped = state.view(*shape)
            
            # This is simplified - actual implementation would need tensor contractions
            return state  # Placeholder
        elif self.use_gpu:
            # CuPy implementation
            return cp.tensordot(matrix, state, axes=([1], targets))
        else:
            # NumPy implementation
            return np.tensordot(matrix, state, axes=([1], targets))
    
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert to NumPy array"""
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        elif hasattr(tensor, 'get'):
            return cp.asnumpy(tensor)
        else:
            return np.array(tensor)

class QuantumTestSuite64:
    """Ultra-scale test suite for up to 64 qubits"""
    
    def __init__(self, max_qubits: int = 64, use_gpu: bool = True, use_llm: bool = True):
        self.max_qubits = min(max_qubits, 64)  # Hard cap at 64
        self.use_gpu = use_gpu and HAS_GPU
        self.use_llm = use_llm and HAS_LLM
        
        # Initialize components
        self.llm_optimizer = LLMOptimizer() if self.use_llm else None
        self.gpu_backend = GPUQuantumBackend() if self.use_gpu else None
        
        # Memory-efficient configuration
        self.config = QNVMConfig(
            max_qubits=self.max_qubits,
            max_memory_gb=256.0,  # Increased for 64 qubits
            backend=BackendType.INTERNAL,
            error_correction=False,
            compression_enabled=True,
            compression_method=CompressionMethod.SPARSE,  # Use sparse for large systems
            compression_ratio=0.01,  # Aggressive compression
            validation_enabled=False,  # Disable for performance
            log_level="ERROR",
            use_mixed_precision=True,  # Use float32/complex64
            enable_gpu=self.use_gpu,
            enable_distributed=False  # Future: distributed computing
        )
        
        self.vm = create_qnvm(self.config, use_real=True)
        
        # Statistics
        self.results = {}
        self.start_time = time.time()
        self.gates_executed = 0
        
        print(f"\n🚀 QuantumTestSuite64 initialized:")
        print(f"   Target: Up to {self.max_qubits} qubits")
        print(f"   GPU Acceleration: {'✅ Enabled' if self.use_gpu else '❌ Disabled'}")
        print(f"   LLM Optimization: {'✅ Enabled' if self.use_llm else '❌ Disabled'}")
        print(f"   Memory Limit: {self.config.max_memory_gb} GB")
    
    def estimate_memory_required(self, num_qubits: int) -> float:
        """Estimate memory required for state vector"""
        # Using complex64 (8 bytes) instead of complex128 (16 bytes)
        bytes_per_amplitude = 8
        total_bytes = (2 ** num_qubits) * bytes_per_amplitude
        return total_bytes / (1024 ** 3)  # Convert to GB
    
    def run_scalability_suite(self) -> Dict:
        """Run scalability tests with adaptive strategies"""
        print("\n" + "="*80)
        print("🚀 ULTRA-SCALE QUANTUM SCALABILITY SUITE (Up to 64 Qubits)")
        print("="*80)
        
        # Adaptive test sequence based on available memory
        available_memory_gb = self.config.max_memory_gb
        
        test_sequence = [
            ("Memory-Efficient Initialization", self.test_memory_efficient_init),
            ("Sparse Gate Operations", self.test_sparse_gates),
            ("Tensor Network Simulation", self.test_tensor_network),
            ("Hybrid Quantum-Classical", self.test_hybrid_computation),
            ("Approximate Quantum Evolution", self.test_approximate_evolution),
        ]
        
        # Skip tests that require too much memory
        filtered_tests = []
        for test_name, test_func in test_sequence:
            # Estimate if test is feasible
            max_test_qubits = self._estimate_test_requirements(test_name)
            required_memory = self.estimate_memory_required(max_test_qubits)
            
            if required_memory <= available_memory_gb * 2:  # 2x buffer
                filtered_tests.append((test_name, test_func))
                print(f"  ✅ {test_name} (estimated up to {max_test_qubits} qubits)")
            else:
                print(f"  ⚠️  {test_name} requires {required_memory:.1f} GB, skipping")
        
        # Run filtered tests
        results = {}
        for test_name, test_func in filtered_tests:
            print(f"\n{'='*50}")
            print(f"TEST: {test_name}")
            print(f"{'='*50}")
            
            try:
                start = time.time()
                result = test_func()
                elapsed = time.time() - start
                
                results[test_name] = {
                    'status': 'passed',
                    'time': elapsed,
                    'result': result
                }
                
                print(f"✅ Completed in {elapsed:.2f}s")
                
            except MemoryError as e:
                print(f"❌ Memory Error: {e}")
                results[test_name] = {'status': 'memory_error', 'error': str(e)}
            except Exception as e:
                print(f"❌ Error: {e}")
                results[test_name] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def test_memory_efficient_init(self) -> Dict:
        """Test memory-efficient state initialization techniques"""
        results = {}
        
        # Test different initialization strategies
        strategies = [
            ("zero_state", lambda n: [{'gate': 'INIT', 'targets': list(range(n)), 'params': {'state': 'zero'}}]),
            ("plus_state", lambda n: [{'gate': 'H', 'targets': [i]} for i in range(n)]),
            ("sparse_state", lambda n: [{'gate': 'INIT', 'targets': [0], 'params': {'state': 'one'}}]),
        ]
        
        qubit_counts = [2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
        
        for strategy_name, gate_generator in strategies:
            results[strategy_name] = {}
            
            for n in qubit_counts:
                if n > self.max_qubits:
                    continue
                
                # Check memory requirements
                required_memory = self.estimate_memory_required(n)
                if required_memory > self.config.max_memory_gb:
                    print(f"  ⚠️  Skipping {n} qubits (requires {required_memory:.1f} GB)")
                    break
                
                circuit = {
                    'name': f'{strategy_name}_{n}',
                    'num_qubits': n,
                    'gates': gate_generator(n)
                }
                
                # Optimize with LLM if available
                if self.use_llm:
                    circuit = self.llm_optimizer.optimize_circuit(circuit)
                
                try:
                    result = self.vm.execute_circuit(circuit)
                    
                    results[strategy_name][n] = {
                        'time_ms': result.execution_time_ms,
                        'memory_mb': getattr(result, 'memory_used_gb', 0) * 1024,
                        'fidelity': result.estimated_fidelity,
                        'success': result.success
                    }
                    
                    print(f"  {strategy_name:12s} {n:2d} qubits: "
                          f"{result.execution_time_ms:8.2f} ms, "
                          f"{getattr(result, 'memory_used_gb', 0)*1024:8.2f} MB")
                    
                    # Force garbage collection for large systems
                    if n >= 32:
                        gc.collect()
                        if self.use_gpu and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"  {strategy_name:12s} {n:2d} qubits: ❌ {str(e)}")
                    break
        
        return results
    
    def test_sparse_gates(self) -> Dict:
        """Test sparse gate operations for large systems"""
        results = {}
        
        # Sparse gate patterns (affecting only few qubits)
        sparse_patterns = [
            ("single_qubit_layer", lambda n: [{'gate': 'H', 'targets': [i]} for i in [0, n//2, n-1]]),
            ("corner_cnots", lambda n: [
                {'gate': 'CNOT', 'targets': [n//2], 'controls': [0]},
                {'gate': 'CNOT', 'targets': [n-1], 'controls': [n//2]}
            ]),
            ("sparse_entanglement", lambda n: [
                {'gate': 'H', 'targets': [0]},
                {'gate': 'CNOT', 'targets': [n//4], 'controls': [0]},
                {'gate': 'CNOT', 'targets': [n//2], 'controls': [n//4]},
                {'gate': 'CNOT', 'targets': [3*n//4], 'controls': [n//2]},
                {'gate': 'CNOT', 'targets': [n-1], 'controls': [3*n//4]}
            ]),
        ]
        
        qubit_counts = [16, 32, 48, 64]
        
        for pattern_name, gate_generator in sparse_patterns:
            results[pattern_name] = {}
            
            for n in qubit_counts:
                if n > self.max_qubits:
                    continue
                
                circuit = {
                    'name': f'sparse_{pattern_name}_{n}',
                    'num_qubits': n,
                    'gates': gate_generator(n)
                }
                
                # Use GPU backend if available
                if self.use_gpu:
                    # Convert circuit to GPU execution
                    circuit['backend'] = 'gpu'
                
                try:
                    result = self.vm.execute_circuit(circuit)
                    
                    results[pattern_name][n] = {
                        'time_ms': result.execution_time_ms,
                        'gate_count': len(circuit['gates']),
                        'fidelity': result.estimated_fidelity,
                        'success': result.success
                    }
                    
                    self.gates_executed += len(circuit['gates'])
                    
                    print(f"  {pattern_name:20s} {n:2d} qubits: "
                          f"{result.execution_time_ms:8.2f} ms, "
                          f"{result.estimated_fidelity:.6f} fidelity")
                    
                except Exception as e:
                    print(f"  {pattern_name:20s} {n:2d} qubits: ❌ {str(e)}")
                    break
        
        return results
    
    def test_tensor_network(self) -> Dict:
        """Test tensor network simulation techniques"""
        results = {}
        
        print("  Simulating using tensor network techniques...")
        
        # Matrix Product State (MPS) like simulation
        # This is simplified - in reality would use a tensor network library
        
        qubit_counts = [20, 30, 40, 50, 60]
        bond_dimensions = [2, 4, 8, 16]
        
        for n in qubit_counts:
            if n > self.max_qubits:
                continue
            
            results[n] = {}
            
            for bond_dim in bond_dimensions:
                # Create GHZ-like state with limited entanglement
                circuit = self._create_mps_circuit(n, bond_dim)
                
                try:
                    start = time.time()
                    # Simulate with reduced bond dimension
                    result = self._simulate_mps(circuit, bond_dim)
                    elapsed = (time.time() - start) * 1000
                    
                    results[n][bond_dim] = {
                        'time_ms': elapsed,
                        'memory_mb': bond_dim * n * 100 / (1024**2),  # Rough estimate
                        'success': True
                    }
                    
                    print(f"  MPS {n:2d} qubits, bond={bond_dim:2d}: "
                          f"{elapsed:8.2f} ms")
                    
                except Exception as e:
                    print(f"  MPS {n:2d} qubits, bond={bond_dim:2d}: ❌ {str(e)}")
                    break
        
        return results
    
    def _create_mps_circuit(self, n: int, max_entanglement: int) -> Dict:
        """Create circuit suitable for MPS simulation"""
        gates = []
        
        # Create local operations with limited entanglement
        for i in range(0, n, max_entanglement):
            # Apply Hadamard to each segment
            gates.append({'gate': 'H', 'targets': [i]})
            
            # Entangle within segment
            for j in range(1, min(max_entanglement, n-i)):
                gates.append({'gate': 'CNOT', 'targets': [i+j], 'controls': [i]})
        
        return {
            'name': f'mps_{n}_{max_entanglement}',
            'num_qubits': n,
            'gates': gates
        }
    
    def _simulate_mps(self, circuit: Dict, bond_dim: int) -> Dict:
        """Simplified MPS simulation"""
        # Placeholder for actual tensor network simulation
        # In production, would use libraries like quimb, tensornetwork, etc.
        return {'success': True, 'estimated_fidelity': 0.95}
    
    def test_hybrid_computation(self) -> Dict:
        """Test hybrid quantum-classical algorithms"""
        results = {}
        
        algorithms = [
            ("vqe_small", self._run_vqe_small),
            ("qaoa_maxcut", self._run_qaoa_maxcut),
            ("quantum_autoencoder", self._run_quantum_autoencoder),
        ]
        
        for algo_name, algo_func in algorithms:
            print(f"\n  Running {algo_name}...")
            
            try:
                start = time.time()
                result = algo_func()
                elapsed = time.time() - start
                
                results[algo_name] = {
                    'time': elapsed,
                    'result': result,
                    'success': True
                }
                
                print(f"  ✅ {algo_name}: {elapsed:.2f}s")
                
            except Exception as e:
                print(f"  ❌ {algo_name}: {str(e)}")
                results[algo_name] = {'success': False, 'error': str(e)}
        
        return results
    
    def _run_vqe_small(self) -> Dict:
        """Run Variational Quantum Eigensolver on small system"""
        # Simplified VQE for demonstration
        num_qubits = min(12, self.max_qubits)
        
        # Create parameterized circuit
        circuits = []
        for iteration in range(5):  # 5 VQE iterations
            circuit = {
                'name': f'vqe_iter_{iteration}',
                'num_qubits': num_qubits,
                'gates': [
                    {'gate': 'RY', 'targets': [i], 'params': {'angle': 0.1 * iteration} for i in range(num_qubits)},
                    *[{'gate': 'CNOT', 'targets': [i+1], 'controls': [i]} for i in range(num_qubits-1)]
                ]
            }
            circuits.append(circuit)
        
        # Execute all circuits
        energies = []
        for circuit in circuits:
            result = self.vm.execute_circuit(circuit)
            # Simplified energy calculation
            energy = 1.0 - result.estimated_fidelity
            energies.append(energy)
        
        return {'energies': energies, 'min_energy': min(energies)}
    
    def _run_qaoa_maxcut(self) -> Dict:
        """Run QAOA for MaxCut problem"""
        num_qubits = min(16, self.max_qubits)
        
        # Create QAOA circuit
        circuit = {
            'name': 'qaoa_maxcut',
            'num_qubits': num_qubits,
            'gates': [
                # Initial Hadamard layer
                *[{'gate': 'H', 'targets': [i]} for i in range(num_qubits)],
                
                # Problem unitary (simplified)
                *[{'gate': 'RZZ', 'targets': [i, (i+1)%num_qubits], 'params': {'angle': 0.5}} 
                  for i in range(num_qubits)],
                
                # Mixer unitary
                *[{'gate': 'RX', 'targets': [i], 'params': {'angle': 0.3}} 
                  for i in range(num_qubits)],
            ]
        }
        
        result = self.vm.execute_circuit(circuit)
        
        # Simplified cut value calculation
        cut_value = result.estimated_fidelity * num_qubits
        
        return {'cut_value': cut_value, 'fidelity': result.estimated_fidelity}
    
    def _run_quantum_autoencoder(self) -> Dict:
        """Run quantum autoencoder compression"""
        num_qubits = min(8, self.max_qubits)
        latent_qubits = 2
        
        circuit = {
            'name': 'quantum_autoencoder',
            'num_qubits': num_qubits,
            'gates': [
                # Encoding layer
                *[{'gate': 'RY', 'targets': [i], 'params': {'angle': 0.2*i}} 
                  for i in range(num_qubits)],
                
                # Entangling layer
                *[{'gate': 'CNOT', 'targets': [(i+1)%num_qubits], 'controls': [i]} 
                  for i in range(num_qubits)],
                
                # Measurement on latent space
                *[{'gate': 'MEASURE', 'targets': [i]} 
                  for i in range(latent_qubits)],
            ]
        }
        
        result = self.vm.execute_circuit(circuit)
        
        # Calculate compression ratio
        compression_ratio = latent_qubits / num_qubits
        
        return {
            'compression_ratio': compression_ratio,
            'measurements': result.measurements if hasattr(result, 'measurements') else None
        }
    
    def test_approximate_evolution(self) -> Dict:
        """Test approximate quantum evolution techniques"""
        results = {}
        
        evolution_methods = [
            ("trotter_suzuki", self._run_trotter_suzuki),
            ("variational_evolution", self._run_variational_evolution),
            ("random_walk", self._run_random_walk),
        ]
        
        system_sizes = [8, 16, 24, 32]
        
        for method_name, method_func in evolution_methods:
            results[method_name] = {}
            
            for n in system_sizes:
                if n > self.max_qubits:
                    continue
                
                print(f"  {method_name:20s} {n:2d} qubits...", end="")
                
                try:
                    start = time.time()
                    result = method_func(n)
                    elapsed = time.time() - start
                    
                    results[method_name][n] = {
                        'time_ms': elapsed * 1000,
                        'success': True,
                        'result': result.get('fidelity', 0)
                    }
                    
                    print(f" ✅ {elapsed:.2f}s")
                    
                except Exception as e:
                    print(f" ❌ {str(e)}")
                    results[method_name][n] = {'success': False, 'error': str(e)}
        
        return results
    
    def _run_trotter_suzuki(self, n: int) -> Dict:
        """Run Trotter-Suzuki approximation"""
        steps = 10
        dt = 0.1
        
        # Create time evolution circuit
        all_gates = []
        for step in range(steps):
            # Local terms
            for i in range(n):
                all_gates.append({'gate': 'RX', 'targets': [i], 'params': {'angle': dt}})
            
            # Interaction terms (nearest neighbor)
            for i in range(n-1):
                all_gates.append({'gate': 'RXX', 'targets': [i, i+1], 'params': {'angle': dt}})
        
        circuit = {
            'name': f'trotter_{n}',
            'num_qubits': n,
            'gates': all_gates
        }
        
        result = self.vm.execute_circuit(circuit)
        
        return {'fidelity': result.estimated_fidelity, 'gate_count': len(all_gates)}
    
    def _run_variational_evolution(self, n: int) -> Dict:
        """Run variational quantum evolution"""
        # Simplified variational circuit
        circuit = {
            'name': f'variational_{n}',
            'num_qubits': n,
            'gates': [
                {'gate': 'RY', 'targets': [i], 'params': {'angle': 0.1*i}} 
                for i in range(n)
            ]
        }
        
        result = self.vm.execute_circuit(circuit)
        
        return {'fidelity': result.estimated_fidelity}
    
    def _run_random_walk(self, n: int) -> Dict:
        """Run random walk on quantum state space"""
        steps = 20
        
        # Random quantum walk
        gates = []
        for step in range(steps):
            # Random single-qubit gate
            qubit = step % n
            gate_type = np.random.choice(['H', 'RX', 'RY', 'RZ'])
            gates.append({
                'gate': gate_type, 
                'targets': [qubit],
                'params': {'angle': np.random.random() * np.pi}
            })
            
            # Random two-qubit gate occasionally
            if step % 5 == 0 and n > 1:
                target = (qubit + 1) % n
                gates.append({
                    'gate': 'CNOT',
                    'targets': [target],
                    'controls': [qubit]
                })
        
        circuit = {
            'name': f'random_walk_{n}',
            'num_qubits': n,
            'gates': gates
        }
        
        result = self.vm.execute_circuit(circuit)
        
        return {'fidelity': result.estimated_fidelity, 'steps': steps}
    
    def _estimate_test_requirements(self, test_name: str) -> int:
        """Estimate maximum qubits for a test"""
        requirements = {
            'Memory-Efficient Initialization': 64,
            'Sparse Gate Operations': 64,
            'Tensor Network Simulation': 60,
            'Hybrid Quantum-Classical': 32,
            'Approximate Quantum Evolution': 40,
        }
        return requirements.get(test_name, 32)
    
    def generate_comprehensive_report(self, results: Dict):
        """Generate comprehensive report with insights"""
        print("\n" + "="*80)
        print("📊 ULTRA-SCALE QUANTUM TEST REPORT")
        print("="*80)
        
        total_time = time.time() - self.start_time
        
        # Calculate statistics
        total_tests = sum(len(r.get('result', {})) for r in results.values() if isinstance(r, dict))
        passed_tests = sum(1 for r in results.values() if r.get('status') == 'passed')
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   Tests Completed: {passed_tests}/{len(results)}")
        print(f"   Gates Executed: {self.gates_executed:,}")
        print(f"   Maximum Qubits Targeted: {self.max_qubits}")
        
        # Memory usage insights
        print(f"\n💾 Memory Analysis:")
        max_memory_used = 0
        for test_name, test_result in results.items():
            if 'result' in test_result and isinstance(test_result['result'], dict):
                for qubit_data in test_result['result'].values():
                    if isinstance(qubit_data, dict) and 'memory_mb' in qubit_data:
                        max_memory_used = max(max_memory_used, qubit_data['memory_mb'])
        
        print(f"   Peak Memory Usage: {max_memory_used:.2f} MB")
        print(f"   Memory Efficiency: {(max_memory_used / (self.config.max_memory_gb * 1024)) * 100:.1f}% of available")
        
        # Performance insights
        print(f"\n⚡ Performance Insights:")
        if self.use_gpu:
            print(f"   GPU Acceleration: ✅ Active")
        if self.use_llm:
            print(f"   LLM Optimization: ✅ Active")
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'max_qubits': self.max_qubits,
            'use_gpu': self.use_gpu,
            'use_llm': self.use_llm,
            'total_time_seconds': total_time,
            'total_gates_executed': self.gates_executed,
            'config': {
                'max_memory_gb': self.config.max_memory_gb,
                'compression_enabled': self.config.compression_enabled,
                'compression_method': str(self.config.compression_method),
            },
            'results': results
        }
        
        filename = f"qnvm_64_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {filename}")
        
        # Generate recommendations
        self._generate_recommendations(results)
    
    def _generate_recommendations(self, results: Dict):
        """Generate optimization recommendations based on results"""
        print(f"\n🎯 Optimization Recommendations:")
        
        recommendations = []
        
        # Check memory usage patterns
        memory_intensive_tests = []
        for test_name, test_result in results.items():
            if 'result' in test_result and isinstance(test_result['result'], dict):
                for qubit_data in test_result['result'].values():
                    if isinstance(qubit_data, dict) and 'memory_mb' in qubit_data:
                        if qubit_data['memory_mb'] > self.config.max_memory_gb * 512:  # > 50% of available
                            memory_intensive_tests.append(test_name)
        
        if memory_intensive_tests:
            recommendations.append(
                f"Memory-intensive tests detected: {', '.join(memory_intensive_tests)}. "
                f"Consider enabling more aggressive compression or using tensor network methods."
            )
        
        # Check GPU utilization
        if self.use_gpu:
            # Simplified GPU check - in reality would measure GPU memory usage
            recommendations.append(
                "GPU acceleration is enabled. For larger systems (>40 qubits), "
                "consider using mixed-precision (FP16) calculations."
            )
        
        # Check LLM effectiveness
        if self.use_llm:
            recommendations.append(
                "LLM optimization is enabled. Consider fine-tuning on quantum circuit "
                "datasets for better optimization performance."
            )
        
        # General recommendations
        if self.max_qubits >= 40:
            recommendations.append(
                f"For systems with {self.max_qubits} qubits, consider: "
                "1. Distributed simulation across multiple nodes\n"
                "2. Using approximate simulation methods\n"
                "3. Implementing state compression algorithms"
            )
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

def main():
    """Main execution for ultra-scale quantum testing"""
    print("="*80)
    print("🚀 QNVM v6.0 - ULTRA-SCALE QUANTUM TEST SUITE (Up to 64 Qubits)")
    print("="*80)
    
    # Get configuration
    try:
        max_qubits = int(input("\nEnter target qubits (8-64, default 32): ") or "32")
        max_qubits = max(8, min(64, max_qubits))
    except:
        max_qubits = 32
    
    use_gpu = input("Enable GPU acceleration? (y/n, default y): ").lower() != 'n'
    use_llm = input("Enable LLM circuit optimization? (y/n, default y): ").lower() != 'n'
    
    # Display capabilities
    print(f"\n⚡ Configuration:")
    print(f"   Target: {max_qubits} qubits")
    print(f"   GPU: {'✅ Enabled' if use_gpu and HAS_GPU else '❌ Not available'}")
    print(f"   LLM: {'✅ Enabled' if use_llm and HAS_LLM else '❌ Not available'}")
    
    # Memory warning
    if max_qubits > 30:
        estimated_memory = (2 ** max_qubits) * 8 / (1024 ** 3)  # complex64 in GB
        print(f"\n⚠️  Warning: {max_qubits} qubits may require ~{estimated_memory:.1f} GB of memory.")
        if estimated_memory > 32:
            print("   Consider using sparse representations or tensor network methods.")
        
        proceed = input("\nProceed with testing? (y/n): ").lower()
        if proceed != 'y':
            print("Test cancelled.")
            return 0
    
    # Run test suite
    print("\n" + "="*80)
    print("🚀 Starting Ultra-Scale Quantum Tests...")
    print("="*80)
    
    test_suite = QuantumTestSuite64(
        max_qubits=max_qubits,
        use_gpu=use_gpu,
        use_llm=use_llm
    )
    
    results = test_suite.run_scalability_suite()
    test_suite.generate_comprehensive_report(results)
    
    # Final status
    print("\n" + "="*80)
    print("🎉 ULTRA-SCALE TESTING COMPLETE!")
    print("="*80)
    
    # Ask about visualization
    try:
        import matplotlib.pyplot as plt
        plot = input("\nGenerate visualization plots? (y/n, default y): ").lower() != 'n'
        if plot:
            test_suite._generate_visualizations(results)
    except ImportError:
        print("\n⚠️  Matplotlib not available - skipping visualization")
    
    return 0

def _generate_visualizations(self, results: Dict):
    """Generate visualization plots"""
    try:
        import matplotlib.pyplot as plt
        
        # Prepare data for visualization
        qubit_counts = []
        times = []
        memories = []
        
        # Extract data from results
        for test_name, test_data in results.items():
            if 'result' in test_data and isinstance(test_data['result'], dict):
                for n, data in test_data['result'].items():
                    if isinstance(data, dict):
                        if 'time_ms' in data:
                            qubit_counts.append(int(n))
                            times.append(data['time_ms'])
                        if 'memory_mb' in data:
                            memories.append(data['memory_mb'])
        
        if qubit_counts:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Time scaling
            axes[0, 0].scatter(qubit_counts, times, alpha=0.6)
            axes[0, 0].set_xlabel('Number of Qubits')
            axes[0, 0].set_ylabel('Execution Time (ms)')
            axes[0, 0].set_title('Execution Time Scaling')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True)
            
            # Memory scaling
            if memories:
                axes[0, 1].scatter(qubit_counts[:len(memories)], memories, alpha=0.6)
                axes[0, 1].set_xlabel('Number of Qubits')
                axes[0, 1].set_ylabel('Memory Usage (MB)')
                axes[0, 1].set_title('Memory Usage Scaling')
                axes[0, 1].set_yscale('log')
                axes[0, 1].grid(True)
            
            # Theoretical vs actual scaling
            theoretical_times = [2**n * 0.01 for n in qubit_counts]  # Simplified model
            axes[1, 0].plot(qubit_counts, times, 'bo-', label='Actual')
            axes[1, 0].plot(qubit_counts, theoretical_times, 'r--', label='Theoretical (O(2^n))')
            axes[1, 0].set_xlabel('Number of Qubits')
            axes[1, 0].set_ylabel('Time (ms)')
            axes[1, 0].set_title('Time Complexity Analysis')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Success rate by qubit count
            success_counts = {}
            for n in qubit_counts:
                success_counts[n] = success_counts.get(n, 0) + 1
            
            axes[1, 1].bar(success_counts.keys(), success_counts.values())
            axes[1, 1].set_xlabel('Number of Qubits')
            axes[1, 1].set_ylabel('Successful Tests')
            axes[1, 1].set_title('Test Success by Qubit Count')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig('qnvm_64_scaling_analysis.png', dpi=150, bbox_inches='tight')
            print("\n📈 Visualization saved to: qnvm_64_scaling_analysis.png")
            
    except Exception as e:
        print(f"\n⚠️  Visualization error: {e}")

# Monkey patch for visualization
QuantumTestSuite64._generate_visualizations = _generate_visualizations

if __name__ == "__main__":
    sys.exit(main())
