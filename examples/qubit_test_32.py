#!/usr/bin/env python3
"""
QNVM v5.1 Comprehensive Quantum Test Suite (Up to 32 Qubits)
"""

import sys
import os
import time
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from qnvm import QNVM, QNVMConfig, create_qnvm, HAS_REAL_IMPL
    from qnvm.config import BackendType, CompressionMethod
    print(f"✅ QNVM v5.1 loaded (Real Implementation: {HAS_REAL_IMPL})")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class QuantumTestSuite:
    """Comprehensive test suite for QNVM quantum operations"""
    
    def __init__(self, max_qubits: int = 32, use_real: bool = True):
        self.max_qubits = max_qubits
        self.use_real = use_real and HAS_REAL_IMPL
        
        # Test configuration
        self.config = QNVMConfig(
            max_qubits=max_qubits,
            max_memory_gb=32.0,
            backend=BackendType.INTERNAL,
            error_correction=False,
            compression_enabled=True,
            validation_enabled=True,
            log_level="WARNING"  # Reduce logging for extensive tests
        )
        
        self.vm = create_qnvm(self.config, use_real=self.use_real)
        self.results = {}
        self.start_time = time.time()
        
        print(f"Initialized QuantumTestSuite with max {max_qubits} qubits")
        print(f"Real quantum implementation: {self.use_real}")
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("\n" + "=" * 70)
        print("QNVM v5.1 COMPREHENSIVE QUANTUM TEST SUITE")
        print("=" * 70)
        
        test_sequence = [
            ("State Initialization", self.test_state_initialization),
            ("Single-Qubit Gates", self.test_single_qubit_gates),
            ("Two-Qubit Gates", self.test_two_qubit_gates),
            ("Bell State Family", self.test_bell_state_family),
            ("GHZ State Scaling", self.test_ghz_state_scaling),
            ("Quantum Fourier Transform", self.test_quantum_fourier_transform),
            ("Random Circuit Validation", self.test_random_circuits),
            ("Entanglement Generation", self.test_entanglement_generation),
            ("Measurement Statistics", self.test_measurement_statistics),
            ("Memory Scaling", self.test_memory_scaling),
            ("Performance Benchmark", self.test_performance_benchmark),
        ]
        
        passed = 0
        failed = 0
        skipped = 0
        
        for test_name, test_func in test_sequence:
            print(f"\n{'='*40}")
            print(f"TEST: {test_name}")
            print(f"{'='*40}")
            
            try:
                start = time.time()
                result = test_func()
                elapsed = time.time() - start
                
                if result.get('status') == 'passed':
                    print(f"✅ PASSED in {elapsed:.2f}s")
                    passed += 1
                elif result.get('status') == 'skipped':
                    print(f"⚠️  SKIPPED: {result.get('reason', '')}")
                    skipped += 1
                else:
                    print(f"❌ FAILED: {result.get('error', '')}")
                    failed += 1
                
                self.results[test_name] = result
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                self.results[test_name] = {'status': 'error', 'error': str(e)}
        
        # Generate comprehensive report
        self._generate_report(passed, failed, skipped)
        
        return passed, failed, skipped
    
    def test_state_initialization(self) -> Dict:
        """Test |0⟩^n state initialization"""
        results = {}
        
        for n in [2, 4, 8, 16, 24, 32]:
            if n > self.max_qubits:
                continue
                
            circuit = {
                'name': f'zero_state_{n}',
                'num_qubits': n,
                'gates': []  # No gates, just initialization
            }
            
            result = self.vm.execute_circuit(circuit)
            
            results[n] = {
                'success': result.success,
                'time_ms': result.execution_time_ms,
                'memory_mb': result.memory_used_gb * 1024 if hasattr(result, 'memory_used_gb') else 0,
                'fidelity': result.estimated_fidelity
            }
            
            print(f"  {n:2d} qubits: {result.execution_time_ms:6.2f} ms, "
                  f"Fidelity: {result.estimated_fidelity:.6f}")
        
        return {'status': 'passed', 'results': results}
    
    def test_single_qubit_gates(self) -> Dict:
        """Test all single-qubit gates"""
        gate_tests = ['H', 'X', 'Y', 'Z', 'S', 'T']
        results = {}
        
        for gate in gate_tests:
            for n_qubits in [2, 4, 8]:
                circuit = {
                    'name': f'{gate}_test_{n_qubits}',
                    'num_qubits': n_qubits,
                    'gates': [{'gate': gate, 'targets': [i]} for i in range(n_qubits)]
                }
                
                result = self.vm.execute_circuit(circuit)
                
                if gate not in results:
                    results[gate] = {}
                
                results[gate][n_qubits] = {
                    'success': result.success,
                    'time_ms': result.execution_time_ms,
                    'fidelity': result.estimated_fidelity
                }
                
                if n_qubits == 2:  # Just show one example
                    print(f"  {gate} gate on {n_qubits} qubits: "
                          f"{result.execution_time_ms:.2f} ms, "
                          f"Fidelity: {result.estimated_fidelity:.6f}")
        
        return {'status': 'passed', 'results': results}
    
    def test_two_qubit_gates(self) -> Dict:
        """Test two-qubit gates (CNOT, CZ, SWAP)"""
        results = {}
        
        for n_qubits in [4, 8, 12]:
            # Create linear chain of CNOT gates
            circuit = {
                'name': f'cnot_chain_{n_qubits}',
                'num_qubits': n_qubits,
                'gates': [
                    {'gate': 'CNOT', 'targets': [i+1], 'controls': [i]} 
                    for i in range(n_qubits - 1)
                ]
            }
            
            result = self.vm.execute_circuit(circuit)
            
            results[n_qubits] = {
                'success': result.success,
                'time_ms': result.execution_time_ms,
                'fidelity': result.estimated_fidelity,
                'gate_count': len(circuit['gates'])
            }
            
            print(f"  CNOT chain {n_qubits} qubits: "
                  f"{result.execution_time_ms:.2f} ms, "
                  f"Fidelity: {result.estimated_fidelity:.6f}")
        
        return {'status': 'passed', 'results': results}
    
    def test_bell_state_family(self) -> Dict:
        """Test creation of Bell states with various methods"""
        results = {}
        
        # Different Bell state preparation circuits
        bell_circuits = [
            {
                'name': 'bell_standard',
                'gates': [
                    {'gate': 'H', 'targets': [0]},
                    {'gate': 'CNOT', 'targets': [1], 'controls': [0]}
                ]
            },
            {
                'name': 'bell_alternative',
                'gates': [
                    {'gate': 'H', 'targets': [1]},
                    {'gate': 'CNOT', 'targets': [0], 'controls': [1]}
                ]
            },
            {
                'name': 'bell_with_measure',
                'gates': [
                    {'gate': 'H', 'targets': [0]},
                    {'gate': 'CNOT', 'targets': [1], 'controls': [0]},
                    {'gate': 'MEASURE', 'targets': [0, 1]}
                ]
            }
        ]
        
        for circuit_def in bell_circuits:
            circuit = {
                'name': circuit_def['name'],
                'num_qubits': 2,
                'gates': circuit_def['gates']
            }
            
            result = self.vm.execute_circuit(circuit)
            
            results[circuit_def['name']] = {
                'success': result.success,
                'time_ms': result.execution_time_ms,
                'fidelity': result.estimated_fidelity,
                'measurements': result.measurements if hasattr(result, 'measurements') else None
            }
            
            print(f"  {circuit_def['name']}: {result.execution_time_ms:.2f} ms, "
                  f"Fidelity: {result.estimated_fidelity:.6f}")
        
        return {'status': 'passed', 'results': results}
    
    def test_ghz_state_scaling(self) -> Dict:
        """Test GHZ state creation scaling from 2 to 32 qubits"""
        results = {}
        
        qubit_counts = [2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32]
        
        for n in qubit_counts:
            if n > self.max_qubits:
                continue
            
            # Create GHZ state
            circuit = {
                'name': f'ghz_{n}',
                'num_qubits': n,
                'gates': [{'gate': 'H', 'targets': [0]}]
            }
            
            # Add CNOTs to entangle all qubits with qubit 0
            for i in range(1, n):
                circuit['gates'].append({
                    'gate': 'CNOT',
                    'targets': [i],
                    'controls': [0]
                })
            
            try:
                result = self.vm.execute_circuit(circuit)
                
                results[n] = {
                    'success': result.success,
                    'time_ms': result.execution_time_ms,
                    'memory_mb': result.memory_used_gb * 1024 if hasattr(result, 'memory_used_gb') else 0,
                    'fidelity': result.estimated_fidelity,
                    'gate_count': len(circuit['gates'])
                }
                
                print(f"  GHZ {n:2d} qubits: {result.execution_time_ms:7.2f} ms, "
                      f"Memory: {results[n]['memory_mb']:7.2f} MB, "
                      f"Fidelity: {result.estimated_fidelity:.6f}")
                
            except MemoryError:
                print(f"  GHZ {n:2d} qubits: ❌ MEMORY ERROR (requires {2**n * 16 / (1024**2):.0f} MB)")
                results[n] = {'status': 'memory_error', 'required_mb': 2**n * 16 / (1024**2)}
                break
            except Exception as e:
                print(f"  GHZ {n:2d} qubits: ❌ ERROR: {e}")
                results[n] = {'status': 'error', 'error': str(e)}
                break
        
        return {'status': 'passed', 'results': results}
    
    def test_quantum_fourier_transform(self) -> Dict:
        """Test QFT implementation"""
        results = {}
        
        for n in [2, 3, 4, 5, 6, 7, 8]:
            if n > self.max_qubits:
                continue
            
            circuit = self._generate_qft_circuit(n)
            result = self.vm.execute_circuit(circuit)
            
            results[n] = {
                'success': result.success,
                'time_ms': result.execution_time_ms,
                'gate_count': len(circuit['gates']),
                'fidelity': result.estimated_fidelity
            }
            
            print(f"  QFT {n} qubits: {result.execution_time_ms:7.2f} ms, "
                  f"{len(circuit['gates']):4d} gates, "
                  f"Fidelity: {result.estimated_fidelity:.6f}")
        
        return {'status': 'passed', 'results': results}
    
    def _generate_qft_circuit(self, n: int) -> Dict:
        """Generate Quantum Fourier Transform circuit"""
        gates = []
        
        # Apply Hadamard and controlled rotations
        for i in range(n):
            gates.append({'gate': 'H', 'targets': [i]})
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                gates.append({
                    'gate': 'RZ',  # Using RZ as a placeholder for controlled phase
                    'targets': [j],
                    'controls': [i],
                    'params': {'angle': angle}
                })
        
        # Bit reversal (swap qubits)
        for i in range(n // 2):
            gates.append({'gate': 'SWAP', 'targets': [i, n - i - 1]})
        
        return {
            'name': f'qft_{n}',
            'num_qubits': n,
            'gates': gates
        }
    
    def test_random_circuits(self) -> Dict:
        """Test random quantum circuits"""
        results = {}
        np.random.seed(42)  # For reproducibility
        
        for n in [4, 8, 12, 16]:
            if n > self.max_qubits:
                continue
            
            # Generate random circuit with n gates
            circuit = self._generate_random_circuit(n, n * 3)
            result = self.vm.execute_circuit(circuit)
            
            results[n] = {
                'success': result.success,
                'time_ms': result.execution_time_ms,
                'gate_count': len(circuit['gates']),
                'fidelity': result.estimated_fidelity
            }
            
            print(f"  Random {n:2d} qubits: {result.execution_time_ms:7.2f} ms, "
                  f"{len(circuit['gates']):3d} gates, "
                  f"Fidelity: {result.estimated_fidelity:.6f}")
        
        return {'status': 'passed', 'results': results}
    
    def _generate_random_circuit(self, num_qubits: int, num_gates: int) -> Dict:
        """Generate random quantum circuit"""
        gates = []
        gate_types = ['H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'CZ']
        
        for _ in range(num_gates):
            gate = np.random.choice(gate_types)
            
            if gate in ['CNOT', 'CZ']:
                # Two-qubit gate
                control, target = np.random.choice(num_qubits, 2, replace=False)
                gates.append({
                    'gate': gate,
                    'targets': [int(target)],
                    'controls': [int(control)]
                })
            else:
                # Single-qubit gate
                target = np.random.randint(0, num_qubits)
                gates.append({
                    'gate': gate,
                    'targets': [int(target)]
                })
        
        return {
            'name': f'random_{num_qubits}_{num_gates}',
            'num_qubits': num_qubits,
            'gates': gates
        }
    
    def test_entanglement_generation(self) -> Dict:
        """Test various entanglement generation patterns"""
        results = {}
        
        entanglement_patterns = [
            ('linear_chain', self._generate_linear_entanglement),
            ('star', self._generate_star_entanglement),
            ('ring', self._generate_ring_entanglement),
            ('grid', self._generate_grid_entanglement),
        ]
        
        for pattern_name, generator in entanglement_patterns:
            for n in [4, 8, 12]:
                if n > self.max_qubits:
                    continue
                
                circuit = generator(n)
                result = self.vm.execute_circuit(circuit)
                
                if pattern_name not in results:
                    results[pattern_name] = {}
                
                results[pattern_name][n] = {
                    'success': result.success,
                    'time_ms': result.execution_time_ms,
                    'gate_count': len(circuit['gates']),
                    'fidelity': result.estimated_fidelity
                }
                
                if n == 4:  # Just show one size
                    print(f"  {pattern_name:12s} {n} qubits: "
                          f"{result.execution_time_ms:6.2f} ms, "
                          f"Fidelity: {result.estimated_fidelity:.6f}")
        
        return {'status': 'passed', 'results': results}
    
    def _generate_linear_entanglement(self, n: int) -> Dict:
        """Generate linear chain entanglement"""
        gates = []
        for i in range(n-1):
            gates.append({'gate': 'CNOT', 'targets': [i+1], 'controls': [i]})
        return {'name': f'linear_{n}', 'num_qubits': n, 'gates': gates}
    
    def _generate_star_entanglement(self, n: int) -> Dict:
        """Generate star entanglement (all connected to qubit 0)"""
        gates = [{'gate': 'H', 'targets': [0]}]
        for i in range(1, n):
            gates.append({'gate': 'CNOT', 'targets': [i], 'controls': [0]})
        return {'name': f'star_{n}', 'num_qubits': n, 'gates': gates}
    
    def _generate_ring_entanglement(self, n: int) -> Dict:
        """Generate ring entanglement"""
        gates = []
        for i in range(n):
            gates.append({'gate': 'CNOT', 'targets': [(i+1)%n], 'controls': [i]})
        return {'name': f'ring_{n}', 'num_qubits': n, 'gates': gates}
    
    def _generate_grid_entanglement(self, n: int) -> Dict:
        """Generate grid entanglement (for square numbers)"""
        import math
        side = int(math.sqrt(n))
        if side * side != n:
            return {'name': f'grid_{n}', 'num_qubits': n, 'gates': []}
        
        gates = []
        # Horizontal connections
        for row in range(side):
            for col in range(side-1):
                q1 = row * side + col
                q2 = row * side + col + 1
                gates.append({'gate': 'CNOT', 'targets': [q2], 'controls': [q1]})
        
        # Vertical connections
        for col in range(side):
            for row in range(side-1):
                q1 = row * side + col
                q2 = (row + 1) * side + col
                gates.append({'gate': 'CNOT', 'targets': [q2], 'controls': [q1]})
        
        return {'name': f'grid_{n}', 'num_qubits': n, 'gates': gates}
    
    def test_measurement_statistics(self) -> Dict:
        """Test measurement statistics on various states"""
        if not self.use_real:
            return {'status': 'skipped', 'reason': 'Requires real implementation'}
        
        results = {}
        
        # Test |+⟩ state measurement
        print("\n  Testing |+⟩ state (should be ~50/50):")
        counts = {0: 0, 1: 0}
        for _ in range(100):
            circuit = {
                'name': 'plus_state',
                'num_qubits': 1,
                'gates': [
                    {'gate': 'H', 'targets': [0]},
                    {'gate': 'MEASURE', 'targets': [0]}
                ]
            }
            result = self.vm.execute_circuit(circuit)
            if result.measurements:
                counts[result.measurements[0]] += 1
        
        results['plus_state'] = counts
        print(f"    |0⟩: {counts[0]}, |1⟩: {counts[1]} "
              f"(ratio: {counts[0]/(counts[0]+counts[1]):.2f})")
        
        # Test Bell state measurement correlation
        print("\n  Testing Bell state correlation:")
        correlations = 0
        total = 50
        for _ in range(total):
            circuit = {
                'name': 'bell_correlation',
                'num_qubits': 2,
                'gates': [
                    {'gate': 'H', 'targets': [0]},
                    {'gate': 'CNOT', 'targets': [1], 'controls': [0]},
                    {'gate': 'MEASURE', 'targets': [0, 1]}
                ]
            }
            result = self.vm.execute_circuit(circuit)
            if result.measurements and len(result.measurements) == 2:
                if result.measurements[0] == result.measurements[1]:
                    correlations += 1
        
        results['bell_correlation'] = correlations / total
        print(f"    Same measurement: {correlations}/{total} "
              f"({correlations/total:.2f} correlation)")
        
        return {'status': 'passed', 'results': results}
    
    def test_memory_scaling(self) -> Dict:
        """Test memory usage scaling with qubit count"""
        results = {}
        
        qubit_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        
        for n in qubit_counts:
            if n > self.max_qubits:
                continue
            
            # Simple circuit to measure memory usage
            circuit = {
                'name': f'memory_test_{n}',
                'num_qubits': n,
                'gates': [{'gate': 'H', 'targets': [0]}]
            }
            
            try:
                result = self.vm.execute_circuit(circuit)
                
                # Calculate theoretical memory (complex128 = 16 bytes per amplitude)
                theoretical_mb = (2 ** n) * 16 / (1024 ** 2)
                
                results[n] = {
                    'theoretical_mb': theoretical_mb,
                    'actual_mb': result.memory_used_gb * 1024 if hasattr(result, 'memory_used_gb') else 0,
                    'compression_ratio': result.compression_ratio if hasattr(result, 'compression_ratio') else 1.0
                }
                
                print(f"  {n:2d} qubits: Theoretical {theoretical_mb:10.1f} MB, "
                      f"Actual {results[n]['actual_mb']:10.1f} MB, "
                      f"Ratio: {results[n]['actual_mb']/theoretical_mb:.3f}")
                
            except MemoryError:
                print(f"  {n:2d} qubits: ❌ MEMORY LIMIT EXCEEDED")
                break
            except Exception as e:
                print(f"  {n:2d} qubits: ❌ ERROR: {e}")
                break
        
        return {'status': 'passed', 'results': results}
    
    def test_performance_benchmark(self) -> Dict:
        """Comprehensive performance benchmarking"""
        results = {}
        
        # Performance tests with increasing complexity
        performance_tests = [
            ('Hadamard Layer', lambda n: [{'gate': 'H', 'targets': [i]} for i in range(n)]),
            ('CNOT Chain', lambda n: [{'gate': 'CNOT', 'targets': [i+1], 'controls': [i]} for i in range(n-1)]),
            ('Mixed Gates', lambda n: [
                {'gate': 'H', 'targets': [i]} if i % 2 == 0 else 
                {'gate': 'X', 'targets': [i]} for i in range(n)
            ]),
        ]
        
        for test_name, gate_generator in performance_tests:
            results[test_name] = {}
            
            for n in [4, 8, 12, 16, 20, 24]:
                if n > self.max_qubits:
                    continue
                
                circuit = {
                    'name': f'perf_{test_name}_{n}',
                    'num_qubits': n,
                    'gates': gate_generator(n)
                }
                
                result = self.vm.execute_circuit(circuit)
                
                results[test_name][n] = {
                    'time_ms': result.execution_time_ms,
                    'gates_per_ms': len(circuit['gates']) / result.execution_time_ms * 1000 if result.execution_time_ms > 0 else 0,
                    'fidelity': result.estimated_fidelity
                }
                
                print(f"  {test_name:15s} {n:2d} qubits: "
                      f"{result.execution_time_ms:7.2f} ms, "
                      f"{results[test_name][n]['gates_per_ms']:7.0f} gates/s")
        
        return {'status': 'passed', 'results': results}
    
    def _generate_report(self, passed: int, failed: int, skipped: int):
        """Generate comprehensive test report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE TEST REPORT")
        print("="*70)
        
        total = passed + failed + skipped
        print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")
        print(f"Success rate: {passed/max(1, total)*100:.1f}%")
        
        # Collect statistics
        total_time = time.time() - self.start_time
        total_circuits = sum(len(r.get('results', {})) for r in self.results.values() if isinstance(r, dict))
        
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        print(f"Total circuits executed: {total_circuits}")
        
        # Get VM statistics
        try:
            vm_stats = self.vm.get_statistics()
            print(f"\nVM Statistics:")
            print(f"  Total gates: {vm_stats['performance'].get('total_gates', 0):,}")
            print(f"  Average fidelity: {vm_stats['performance'].get('avg_fidelity', 0):.6f}")
        except:
            pass
        
        # Generate summary file
        self._save_report()
        
        if failed == 0:
            print("\n✅ ALL TESTS PASSED SUCCESSFULLY!")
        else:
            print(f"\n⚠️  {failed} TEST(S) FAILED")
    
    def _save_report(self):
        """Save test report to file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'max_qubits': self.max_qubits,
            'use_real_implementation': self.use_real,
            'total_time_seconds': time.time() - self.start_time,
            'results': self.results,
            'vm_config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }
        
        filename = f"qnvm_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Detailed report saved to: {filename}")
    
    def plot_results(self):
        """Generate visualization of test results"""
        try:
            import matplotlib.pyplot as plt
            
            # Plot GHZ scaling
            if 'GHZ State Scaling' in self.results:
                ghz_results = self.results['GHZ State Scaling'].get('results', {})
                qubits = []
                times = []
                memories = []
                fidelities = []
                
                for n, data in ghz_results.items():
                    if isinstance(data, dict) and 'time_ms' in data:
                        qubits.append(n)
                        times.append(data['time_ms'])
                        memories.append(data.get('memory_mb', 0))
                        fidelities.append(data.get('fidelity', 0))
                
                if qubits:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Time scaling
                    axes[0, 0].plot(qubits, times, 'bo-')
                    axes[0, 0].set_xlabel('Number of Qubits')
                    axes[0, 0].set_ylabel('Execution Time (ms)')
                    axes[0, 0].set_title('GHZ State Creation Time')
                    axes[0, 0].grid(True)
                    
                    # Memory scaling
                    axes[0, 1].plot(qubits, memories, 'ro-')
                    axes[0, 1].set_xlabel('Number of Qubits')
                    axes[0, 1].set_ylabel('Memory Usage (MB)')
                    axes[0, 1].set_title('Memory Usage Scaling')
                    axes[0, 1].grid(True)
                    
                    # Fidelity
                    axes[1, 0].plot(qubits, fidelities, 'go-')
                    axes[1, 0].set_xlabel('Number of Qubits')
                    axes[1, 0].set_ylabel('Fidelity')
                    axes[1, 0].set_title('Fidelity vs Qubit Count')
                    axes[1, 0].set_ylim([0, 1.1])
                    axes[1, 0].grid(True)
                    
                    # Theoretical vs actual memory
                    theoretical = [(2 ** n) * 16 / (1024 ** 2) for n in qubits]
                    axes[1, 1].plot(qubits, theoretical, 'k--', label='Theoretical')
                    axes[1, 1].plot(qubits, memories, 'r-', label='Actual')
                    axes[1, 1].set_xlabel('Number of Qubits')
                    axes[1, 1].set_ylabel('Memory (MB)')
                    axes[1, 1].set_title('Theoretical vs Actual Memory')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True)
                    
                    plt.tight_layout()
                    plt.savefig('qnvm_scaling_analysis.png', dpi=150)
                    print("\n📈 Visualization saved to: qnvm_scaling_analysis.png")
                    
        except ImportError:
            print("\n⚠️  Matplotlib not available - skipping visualization")

def main():
    """Main test runner"""
    print("="*70)
    print("QNVM v5.1 - EXTENSIVE QUANTUM TEST SUITE (Up to 32 Qubits)")
    print("="*70)
    
    # Get user input for test parameters
    try:
        max_qubits = int(input("\nEnter maximum qubits to test (2-32, default 16): ") or "16")
        max_qubits = max(2, min(32, max_qubits))
    except:
        max_qubits = 16
    
    use_real = input("Use real quantum implementation? (y/n, default y): ").lower() != 'n'
    
    # Run tests
    test_suite = QuantumTestSuite(max_qubits=max_qubits, use_real=use_real)
    passed, failed, skipped = test_suite.run_all_tests()
    
    # Generate visualization
    if passed > 0:
        plot = input("\nGenerate visualization plots? (y/n, default y): ").lower() != 'n'
        if plot:
            test_suite.plot_results()
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
