"""
Benchmark utilities for QNVM
"""

import numpy as np
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

from .core import QNVM, CircuitResult
from .config import QNVMConfig

@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""
    name: str
    config: Dict
    results: Dict
    metadata: Dict
    timestamp: float
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = []
        for qubits, circuit_results in self.results.items():
            for circuit_type, result in circuit_results.items():
                if isinstance(result, dict) and 'time_ms' in result:
                    data.append({
                        'qubits': qubits,
                        'circuit_type': circuit_type,
                        'time_ms': result['time_ms'],
                        'memory_gb': result.get('memory_gb', 0),
                        'fidelity': result.get('fidelity', 0),
                        'success': result.get('success', False)
                    })
        return pd.DataFrame(data)
    
    def plot(self, save_path: Optional[str] = None):
        """Plot benchmark results"""
        df = self.to_dataframe()
        
        if df.empty:
            print("No benchmark data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Execution time by qubit count
        for circuit_type in df['circuit_type'].unique():
            subset = df[df['circuit_type'] == circuit_type]
            axes[0, 0].plot(subset['qubits'], subset['time_ms'], 
                           marker='o', label=circuit_type)
        
        axes[0, 0].set_xlabel('Number of Qubits')
        axes[0, 0].set_ylabel('Execution Time (ms)')
        axes[0, 0].set_title('Execution Time vs Qubit Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage by qubit count
        for circuit_type in df['circuit_type'].unique():
            subset = df[df['circuit_type'] == circuit_type]
            axes[0, 1].plot(subset['qubits'], subset['memory_gb'], 
                           marker='s', label=circuit_type)
        
        axes[0, 1].set_xlabel('Number of Qubits')
        axes[0, 1].set_ylabel('Memory Usage (GB)')
        axes[0, 1].set_title('Memory Usage vs Qubit Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Fidelity by qubit count
        for circuit_type in df['circuit_type'].unique():
            subset = df[df['circuit_type'] == circuit_type]
            axes[1, 0].plot(subset['qubits'], subset['fidelity'], 
                           marker='^', label=circuit_type)
        
        axes[1, 0].set_xlabel('Number of Qubits')
        axes[1, 0].set_ylabel('Fidelity')
        axes[1, 0].set_title('Fidelity vs Qubit Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate
        success_rate = df.groupby('qubits')['success'].mean()
        axes[1, 1].bar(success_rate.index, success_rate.values)
        axes[1, 1].set_xlabel('Number of Qubits')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Success Rate vs Qubit Count')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

class QuantumBenchmark:
    """Benchmark suite for QNVM"""
    
    def __init__(self, config: Optional[QNVMConfig] = None):
        self.config = config or QNVMConfig()
        self.qnvm = QNVM(self.config)
    
    def run_standard_benchmark(self, 
                              qubit_range: List[int] = [4, 8, 12, 16, 20, 24, 28, 32],
                              circuits_per_size: int = 3) -> BenchmarkResult:
        """
        Run standard benchmark suite
        
        Args:
            qubit_range: List of qubit counts to test
            circuits_per_size: Number of circuits to run per qubit count
        
        Returns:
            BenchmarkResult with all results
        """
        results = {}
        
        print("=" * 60)
        print("QNVM Standard Benchmark Suite")
        print("=" * 60)
        
        for num_qubits in qubit_range:
            print(f"\nBenchmarking {num_qubits} qubits...")
            
            circuit_results = {}
            
            # Test different circuit types
            circuit_types = ['ghz', 'qft', 'random', 'vqe']
            
            for circuit_type in circuit_types:
                try:
                    # Generate circuit
                    circuit = self._generate_benchmark_circuit(
                        num_qubits=num_qubits,
                        circuit_type=circuit_type,
                        num_gates=min(100, num_qubits * 10)
                    )
                    
                    # Execute circuit
                    start_time = time.time()
                    result = self.qnvm.execute_circuit(circuit)
                    execution_time = time.time() - start_time
                    
                    circuit_results[circuit_type] = {
                        'time_ms': result.execution_time_ms,
                        'memory_gb': result.memory_used_gb,
                        'fidelity': result.estimated_fidelity,
                        'success': result.success,
                        'validation_passed': result.validation_passed
                    }
                    
                    print(f"  {circuit_type.upper():10s} "
                          f"Time: {result.execution_time_ms:6.2f}ms "
                          f"Memory: {result.memory_used_gb:6.3f}GB "
                          f"Fidelity: {result.estimated_fidelity:.6f}")
                    
                except Exception as e:
                    print(f"  {circuit_type.upper():10s} FAILED: {e}")
                    circuit_results[circuit_type] = {
                        'error': str(e),
                        'success': False
                    }
            
            results[num_qubits] = circuit_results
        
        # Get final statistics
        stats = self.qnvm.get_statistics()
        
        return BenchmarkResult(
            name="standard_benchmark",
            config=self.config.to_dict(),
            results=results,
            metadata={
                'qubit_range': qubit_range,
                'circuits_per_size': circuits_per_size,
                'performance_stats': stats['performance']
            },
            timestamp=time.time()
        )
    
    def run_scaling_analysis(self, 
                            max_qubits: int = 32,
                            step_size: int = 2) -> BenchmarkResult:
        """
        Run scaling analysis to understand performance characteristics
        
        Args:
            max_qubits: Maximum number of qubits to test
            step_size: Step size for qubit count increments
        
        Returns:
            BenchmarkResult with scaling analysis
        """
        qubit_range = list(range(4, max_qubits + 1, step_size))
        results = {}
        
        print("=" * 60)
        print("QNVM Scaling Analysis")
        print("=" * 60)
        
        for num_qubits in qubit_range:
            print(f"\nTesting {num_qubits} qubits...")
            
            # Generate simple GHZ circuit
            circuit = self._generate_benchmark_circuit(
                num_qubits=num_qubits,
                circuit_type='ghz'
            )
            
            # Execute circuit
            result = self.qnvm.execute_circuit(circuit)
            
            results[num_qubits] = {
                'ghz': {
                    'time_ms': result.execution_time_ms,
                    'memory_gb': result.memory_used_gb,
                    'fidelity': result.estimated_fidelity,
                    'success': result.success
                }
            }
            
            print(f"  Time: {result.execution_time_ms:6.2f}ms "
                  f"Memory: {result.memory_used_gb:6.3f}GB "
                  f"Fidelity: {result.estimated_fidelity:.6f}")
        
        # Calculate scaling factors
        scaling_data = self._analyze_scaling(results)
        
        return BenchmarkResult(
            name="scaling_analysis",
            config=self.config.to_dict(),
            results=results,
            metadata={
                'scaling_analysis': scaling_data,
                'max_qubits': max_qubits,
                'step_size': step_size
            },
            timestamp=time.time()
        )
    
    def run_memory_analysis(self) -> BenchmarkResult:
        """Analyze memory usage patterns"""
        print("=" * 60)
        print("QNVM Memory Analysis")
        print("=" * 60)
        
        results = {}
        
        # Test different circuit types
        circuit_types = ['ghz', 'qft', 'random']
        
        for circuit_type in circuit_types:
            print(f"\nAnalyzing {circuit_type} circuits...")
            
            circuit_results = {}
            
            for num_qubits in [8, 16, 24, 32]:
                try:
                    circuit = self._generate_benchmark_circuit(
                        num_qubits=num_qubits,
                        circuit_type=circuit_type,
                        num_gates=50
                    )
                    
                    result = self.qnvm.execute_circuit(circuit)
                    
                    circuit_results[num_qubits] = {
                        'time_ms': result.execution_time_ms,
                        'memory_gb': result.memory_used_gb,
                        'compression_ratio': result.compression_ratio,
                        'state_representation': result.state_representation,
                        'fidelity': result.estimated_fidelity
                    }
                    
                    print(f"  {num_qubits:2d} qubits: "
                          f"Memory: {result.memory_used_gb:6.3f}GB "
                          f"Compression: {result.compression_ratio:.3f} "
                          f"({result.state_representation})")
                    
                except Exception as e:
                    print(f"  {num_qubits:2d} qubits: FAILED - {e}")
                    circuit_results[num_qubits] = {'error': str(e)}
            
            results[circuit_type] = circuit_results
        
        return BenchmarkResult(
            name="memory_analysis",
            config=self.config.to_dict(),
            results=results,
            metadata={
                'circuit_types': circuit_types,
                'memory_requirements': self.config.get_memory_requirements()
            },
            timestamp=time.time()
        )
    
    def _generate_benchmark_circuit(self, 
                                   num_qubits: int,
                                   circuit_type: str,
                                   num_gates: Optional[int] = None) -> Dict:
        """Generate benchmark circuit of specified type"""
        if circuit_type == 'ghz':
            return self._generate_ghz_circuit(num_qubits)
        elif circuit_type == 'qft':
            return self._generate_qft_circuit(num_qubits)
        elif circuit_type == 'random':
            return self._generate_random_circuit(num_qubits, num_gates or 50)
        elif circuit_type == 'vqe':
            return self._generate_vqe_circuit(num_qubits)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    def _generate_ghz_circuit(self, num_qubits: int) -> Dict:
        """Generate GHZ state circuit"""
        gates = []
        
        if num_qubits > 0:
            gates.append({'gate': 'H', 'targets': [0]})
            for i in range(1, num_qubits):
                gates.append({'gate': 'CNOT', 'targets': [i], 'controls': [0]})
        
        return {
            'name': f'ghz_{num_qubits}',
            'num_qubits': num_qubits,
            'type': 'ghz',
            'gates': gates,
            'description': f'GHZ state on {num_qubits} qubits'
        }
    
    def _generate_qft_circuit(self, num_qubits: int) -> Dict:
        """Generate Quantum Fourier Transform circuit"""
        gates = []
        
        for i in range(num_qubits):
            gates.append({'gate': 'H', 'targets': [i]})
            
            for j in range(i + 1, num_qubits):
                angle = np.pi / (2 ** (j - i))
                gates.append({
                    'gate': 'CPHASE',
                    'targets': [j],
                    'controls': [i],
                    'params': {'angle': angle}
                })
        
        # Bit reversal
        for i in range(num_qubits // 2):
            gates.append({'gate': 'SWAP', 'targets': [i, num_qubits - i - 1]})
        
        return {
            'name': f'qft_{num_qubits}',
            'num_qubits': num_qubits,
            'type': 'qft',
            'gates': gates,
            'description': f'QFT on {num_qubits} qubits'
        }
    
    def _generate_random_circuit(self, num_qubits: int, num_gates: int) -> Dict:
        """Generate random quantum circuit"""
        gates = []
        gate_types = ['H', 'X', 'Y', 'Z', 'S', 'T', 'RX', 'RY', 'RZ', 'CNOT', 'CZ']
        
        for i in range(num_gates):
            gate_type = np.random.choice(gate_types)
            
            if gate_type in ['CNOT', 'CZ']:
                # Two-qubit gate
                control, target = np.random.choice(num_qubits, 2, replace=False)
                gates.append({
                    'gate': gate_type,
                    'targets': [int(target)],
                    'controls': [int(control)]
                })
            elif gate_type in ['RX', 'RY', 'RZ']:
                # Parameterized single-qubit gate
                target = np.random.randint(0, num_qubits)
                angle = np.random.uniform(0, 2*np.pi)
                gates.append({
                    'gate': gate_type,
                    'targets': [int(target)],
                    'params': {'angle': angle}
                })
            else:
                # Simple single-qubit gate
                target = np.random.randint(0, num_qubits)
                gates.append({
                    'gate': gate_type,
                    'targets': [int(target)]
                })
        
        return {
            'name': f'random_{num_qubits}_{num_gates}',
            'num_qubits': num_qubits,
            'type': 'random',
            'gates': gates,
            'description': f'Random circuit with {num_gates} gates'
        }
    
    def _generate_vqe_circuit(self, num_qubits: int) -> Dict:
        """Generate VQE ansatz circuit"""
        gates = []
        
        # Alternating layers of single-qubit rotations and entangling gates
        num_layers = 3
        
        for layer in range(num_layers):
            # Single-qubit rotations
            for qubit in range(num_qubits):
                gates.append({
                    'gate': 'RY',
                    'targets': [qubit],
                    'params': {'angle': np.pi/4}  # Fixed for benchmark
                })
            
            # Entangling layer (nearest neighbor)
            for qubit in range(0, num_qubits-1, 2):
                gates.append({
                    'gate': 'CNOT',
                    'targets': [qubit+1],
                    'controls': [qubit]
                })
        
        return {
            'name': f'vqe_{num_qubits}',
            'num_qubits': num_qubits,
            'type': 'vqe',
            'gates': gates,
            'description': f'VQE ansatz with {num_layers} layers'
        }
    
    def _analyze_scaling(self, results: Dict) -> Dict:
        """Analyze scaling characteristics"""
        scaling_data = {}
        
        for qubits, circuit_results in results.items():
            if 'ghz' in circuit_results and circuit_results['ghz']['success']:
                result = circuit_results['ghz']
                
                # Calculate scaling ratios
                if qubits > 4:
                    prev_qubits = qubits - 4
                    if prev_qubits in results and 'ghz' in results[prev_qubits]:
                        prev_result = results[prev_qubits]['ghz']
                        
                        time_scaling = result['time_ms'] / prev_result['time_ms']
                        memory_scaling = result['memory_gb'] / prev_result['memory_gb']
                        
                        scaling_data[qubits] = {
                            'time_scaling': time_scaling,
                            'memory_scaling': memory_scaling,
                            'expected_time_scaling': 2 ** 4,  # 16x for 4 more qubits
                            'expected_memory_scaling': 2 ** 4  # 16x for 4 more qubits
                        }
        
        return scaling_data
    
    def save_benchmark_results(self, result: BenchmarkResult, filename: str):
        """Save benchmark results to file"""
        with open(filename, 'w') as f:
            json.dump({
                'name': result.name,
                'config': result.config,
                'results': result.results,
                'metadata': result.metadata,
                'timestamp': result.timestamp
            }, f, indent=2, default=str)
        
        print(f"Benchmark results saved to {filename}")
    
    @staticmethod
    def load_benchmark_results(filename: str) -> BenchmarkResult:
        """Load benchmark results from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return BenchmarkResult(
            name=data['name'],
            config=data['config'],
            results=data['results'],
            metadata=data['metadata'],
            timestamp=data['timestamp']
        )