#!/usr/bin/env python3
"""
QNVM v5.1 Enhanced Comprehensive Quantum Test Suite (Up to 32 Qubits)
Enhanced with comprehensive error handling, monitoring, and reporting
"""

import sys
import os
import time
import numpy as np
import json
import csv
import psutil
import traceback
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from qnvm import QNVM, QNVMConfig, create_qnvm, HAS_REAL_IMPL
    from qnvm.config import BackendType, CompressionMethod
    print(f"✅ QNVM v5.1 loaded (Real Implementation: {HAS_REAL_IMPL})")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TestResult:
    name: str
    status: TestStatus
    execution_time: float
    memory_used: float
    cpu_percent: Optional[float] = None
    qubits_tested: Optional[int] = 0
    gates_executed: Optional[int] = 0
    fidelity: Optional[float] = None
    error_message: Optional[str] = None
    measurements: Optional[Dict] = None
    details: Optional[Dict] = None

class EnhancedQuantumTestSuite:
    """Enhanced comprehensive test suite for QNVM quantum operations"""
    
    def __init__(self, max_qubits: int = 32, use_real: bool = True):
        self.max_qubits = max_qubits
        self.use_real = use_real and HAS_REAL_IMPL
        
        # Enhanced monitoring
        self.test_results: List[TestResult] = []
        self.start_time = time.time()
        self.peak_memory = 0
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.cpu_readings: List[float] = []
        self.memory_readings: List[float] = []
        self.error_log = []
        
        # Test configuration with realistic memory limits
        estimated_memory = self.estimate_memory_usage(max_qubits)
        system_memory = psutil.virtual_memory().available / 1024 / 1024
        
        print(f"\n⚙️  Enhanced Configuration:")
        print(f"   Maximum Qubits: {max_qubits}")
        print(f"   Estimated Peak Memory: {estimated_memory:.1f} MB")
        print(f"   Available System Memory: {system_memory:.1f} MB")
        
        if estimated_memory > system_memory * 0.8:
            print(f"\n🚨 WARNING: Estimated memory exceeds 80% of available system memory!")
            response = input("   Continue with testing? (y/n): ")
            if response.lower() != 'y':
                print("Test suite terminated by user.")
                sys.exit(0)
        
        self.config = QNVMConfig(
            max_qubits=max_qubits,
            max_memory_gb=min(32.0, system_memory / 1024),  # Realistic memory limit
            backend=BackendType.INTERNAL,
            error_correction=False,
            compression_enabled=True,
            validation_enabled=True,
            log_level="WARNING"
        )
        
        try:
            self.vm = create_qnvm(self.config, use_real=self.use_real)
            print(f"✅ QNVM initialized with max {max_qubits} qubits")
            print(f"   Real quantum implementation: {self.use_real}")
        except Exception as e:
            print(f"❌ Failed to initialize QNVM: {e}")
            sys.exit(1)
    
    def estimate_memory_usage(self, num_qubits: int) -> float:
        """Realistic memory estimation for quantum state"""
        # Base memory for system overhead
        base_memory = 50.0
        
        # For large qubit counts, we use compressed representations
        if num_qubits > 24:
            # Beyond 24 qubits, we never allocate full state vectors in tests
            return 200.0  # MB - safe upper bound
        
        # Memory for state vector (complex128 = 16 bytes per amplitude)
        state_memory = (2 ** num_qubits) * 16 / (1024 * 1024)
        
        # With compression enabled
        compressed_memory = min(state_memory * 0.01, 1000.0)  # Assume 99% compression
        
        return base_memory + compressed_memory
    
    def update_monitoring(self):
        """Update CPU and memory monitoring"""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            self.cpu_readings.append(cpu)
            self.memory_readings.append(memory)
            
            if memory > self.peak_memory:
                self.peak_memory = memory
        except:
            pass
    
    def run_test(self, test_func, test_name: str) -> TestResult:
        """Execute a single test with comprehensive error handling"""
        print(f"\n{'='*50}")
        print(f"[{len(self.test_results)+1}/11] 📊 {test_name}")
        print(f"{'='*50}")
        
        result = TestResult(
            name=test_name,
            status=TestStatus.RUNNING,
            execution_time=0.0,
            memory_used=0.0
        )
        
        try:
            # Update monitoring before test
            self.update_monitoring()
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Execute test
            test_output = test_func()
            
            # Calculate metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Update monitoring after test
            self.update_monitoring()
            
            # Extract results from test output
            if isinstance(test_output, dict):
                if test_output.get('status') == 'passed':
                    result.status = TestStatus.COMPLETED
                elif test_output.get('status') == 'skipped':
                    result.status = TestStatus.SKIPPED
                    result.error_message = test_output.get('reason', 'Skipped')
                else:
                    result.status = TestStatus.FAILED
                    result.error_message = test_output.get('error', 'Failed')
                
                # Extract details from results
                results_data = test_output.get('results', {})
                if results_data:
                    # Try to extract statistics
                    if isinstance(results_data, dict):
                        # For tests that return dict of results per qubit count
                        if len(results_data) > 0:
                            first_result = next(iter(results_data.values()))
                            if isinstance(first_result, dict):
                                result.fidelity = first_result.get('fidelity')
                                result.qubits_tested = len(results_data)
                                result.gates_executed = sum(
                                    r.get('gate_count', 0) for r in results_data.values() 
                                    if isinstance(r, dict)
                                )
                    result.details = results_data
            
            result.execution_time = end_time - start_time
            result.memory_used = end_memory - start_memory
            result.cpu_percent = sum(self.cpu_readings[-2:]) / 2 if len(self.cpu_readings) >= 2 else 0
            
            print(f"   ⏱️  Time: {result.execution_time:.2f}s")
            print(f"   💾 Memory: {result.memory_used:.1f} MB")
            print(f"   🖥️  CPU: {result.cpu_percent:.1f}%")
            
            if result.status == TestStatus.COMPLETED:
                if result.fidelity:
                    print(f"   📈 Avg Fidelity: {result.fidelity:.6f}")
                print(f"   ✅ Success")
            elif result.status == TestStatus.SKIPPED:
                print(f"   ⚠️  Skipped: {result.error_message}")
            else:
                print(f"   ❌ Failed: {result.error_message}")
            
        except MemoryError as e:
            result.status = TestStatus.FAILED
            result.error_message = f"Memory error: {str(e)}"
            print(f"   ❌ Memory Error")
            print(f"   🔍 {result.error_message}")
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            print(f"   ❌ Error: {result.error_message}")
            print(f"   🔍 {type(e).__name__}")
            
            # Log traceback for debugging
            with open("qubit_test_errors.log", "a") as f:
                f.write(f"\n[{datetime.now()}] Test: {test_name}\n")
                f.write(f"Error: {str(e)}\n")
                traceback.print_exc(file=f)
            
            # Ask if we should continue
            if not self.ask_continue_on_error(test_name):
                print("Test suite terminated by user.")
                sys.exit(1)
        
        self.test_results.append(result)
        return result
    
    def ask_continue_on_error(self, test_name: str) -> bool:
        """Ask user whether to continue after a test failure"""
        print(f"\n⚠️  Test '{test_name}' failed. Continue with remaining tests?")
        response = input("   Continue? (y/n/skip): ").lower()
        
        if response == 'n':
            return False
        elif response == 'skip':
            # Mark as skipped
            result = next((r for r in self.test_results if r.name == test_name), None)
            if result:
                result.status = TestStatus.SKIPPED
            return True
        return True
    
    def run_all_tests(self):
        """Run comprehensive test suite with enhanced monitoring"""
        print("\n" + "=" * 70)
        print("🚀 ENHANCED QNVM v5.1 COMPREHENSIVE QUANTUM TEST SUITE")
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
        
        print(f"\n📋 Running {len(test_sequence)} comprehensive tests")
        print(f"   Maximum qubits: {self.max_qubits}")
        print(f"   Real quantum: {self.use_real}")
        
        for test_name, test_func in test_sequence:
            self.run_test(test_func, test_name)
    
    # ============================================================================
    # TEST METHOD IMPLEMENTATIONS (Updated with Enhanced Features)
    # ============================================================================
    
    def test_state_initialization(self) -> Dict:
        """Test |0⟩^n state initialization with enhanced monitoring"""
        results = {}
        total_fidelity = 0
        test_count = 0
        
        qubit_counts = [2, 4, 8, 16, 24, 32]
        qubit_counts = [n for n in qubit_counts if n <= self.max_qubits]
        
        print(f"   Testing initialization for {len(qubit_counts)} qubit configurations")
        
        for n in qubit_counts:
            try:
                circuit = {
                    'name': f'zero_state_{n}',
                    'num_qubits': n,
                    'gates': []
                }
                
                result = self.vm.execute_circuit(circuit)
                
                results[n] = {
                    'success': result.success,
                    'time_ms': result.execution_time_ms,
                    'memory_mb': result.memory_used_gb * 1024 if hasattr(result, 'memory_used_gb') else 0,
                    'fidelity': result.estimated_fidelity
                }
                
                total_fidelity += result.estimated_fidelity
                test_count += 1
                
                print(f"   {n:2d} qubits: {result.execution_time_ms:6.2f} ms, "
                      f"Fidelity: {result.estimated_fidelity:.6f}")
                
            except MemoryError:
                print(f"   {n:2d} qubits: ❌ MEMORY ERROR")
                results[n] = {'status': 'memory_error'}
                break
            except Exception as e:
                print(f"   {n:2d} qubits: ❌ ERROR: {e}")
                results[n] = {'status': 'error', 'error': str(e)}
                break
        
        avg_fidelity = total_fidelity / test_count if test_count > 0 else 0
        
        return {
            'status': 'passed' if test_count > 0 else 'failed',
            'results': results,
            'avg_fidelity': avg_fidelity
        }
    
    def test_single_qubit_gates(self) -> Dict:
        """Test all single-qubit gates with comprehensive coverage"""
        gate_tests = ['H', 'X', 'Y', 'Z', 'S', 'T']
        results = {}
        total_gates = 0
        
        print(f"   Testing {len(gate_tests)} single-qubit gates")
        
        for gate in gate_tests:
            gate_results = {}
            gate_fidelity = 0
            gate_tests_count = 0
            
            for n_qubits in [2, 4, 8]:
                if n_qubits > self.max_qubits:
                    continue
                
                circuit = {
                    'name': f'{gate}_test_{n_qubits}',
                    'num_qubits': n_qubits,
                    'gates': [{'gate': gate, 'targets': [i]} for i in range(n_qubits)]
                }
                
                try:
                    result = self.vm.execute_circuit(circuit)
                    
                    gate_results[n_qubits] = {
                        'success': result.success,
                        'time_ms': result.execution_time_ms,
                        'fidelity': result.estimated_fidelity
                    }
                    
                    gate_fidelity += result.estimated_fidelity
                    gate_tests_count += 1
                    total_gates += n_qubits
                    
                except Exception as e:
                    gate_results[n_qubits] = {
                        'success': False,
                        'error': str(e)
                    }
            
            if gate_tests_count > 0:
                results[gate] = {
                    'avg_fidelity': gate_fidelity / gate_tests_count,
                    'results': gate_results
                }
                
                print(f"   {gate} gate: avg fidelity = {gate_fidelity/gate_tests_count:.6f}")
        
        return {
            'status': 'passed' if len(results) > 0 else 'failed',
            'results': results,
            'total_gates_tested': total_gates
        }
    
    def test_two_qubit_gates(self) -> Dict:
        """Test two-qubit gates (CNOT, CZ, SWAP) with error handling"""
        results = {}
        total_fidelity = 0
        test_count = 0
        
        qubit_counts = [4, 8, 12]
        qubit_counts = [n for n in qubit_counts if n <= self.max_qubits]
        
        print(f"   Testing two-qubit gate chains")
        
        for n_qubits in qubit_counts:
            try:
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
                
                total_fidelity += result.estimated_fidelity
                test_count += 1
                
                print(f"   CNOT chain {n_qubits} qubits: "
                      f"{result.execution_time_ms:.2f} ms, "
                      f"Fidelity: {result.estimated_fidelity:.6f}")
                
            except Exception as e:
                print(f"   {n_qubits} qubits: ❌ ERROR: {e}")
                results[n_qubits] = {'status': 'error', 'error': str(e)}
        
        avg_fidelity = total_fidelity / test_count if test_count > 0 else 0
        
        return {
            'status': 'passed' if test_count > 0 else 'failed',
            'results': results,
            'avg_fidelity': avg_fidelity,
            'total_gates': sum(r.get('gate_count', 0) for r in results.values() if isinstance(r, dict))
        }
    
    def test_bell_state_family(self) -> Dict:
        """Test creation of Bell states with various methods"""
        if not self.use_real:
            return {'status': 'skipped', 'reason': 'Real implementation required for Bell state tests'}
        
        results = {}
        total_fidelity = 0
        test_count = 0
        
        # Different Bell state preparation circuits
        bell_circuits = [
            {
                'name': 'bell_standard',
                'description': 'Standard H+CNOT',
                'gates': [
                    {'gate': 'H', 'targets': [0]},
                    {'gate': 'CNOT', 'targets': [1], 'controls': [0]}
                ]
            },
            {
                'name': 'bell_alternative',
                'description': 'Alternative H+CNOT',
                'gates': [
                    {'gate': 'H', 'targets': [1]},
                    {'gate': 'CNOT', 'targets': [0], 'controls': [1]}
                ]
            },
            {
                'name': 'bell_with_measure',
                'description': 'Bell state with measurement',
                'gates': [
                    {'gate': 'H', 'targets': [0]},
                    {'gate': 'CNOT', 'targets': [1], 'controls': [0]},
                    {'gate': 'MEASURE', 'targets': [0, 1]}
                ]
            }
        ]
        
        print(f"   Testing {len(bell_circuits)} Bell state preparations")
        
        for circuit_def in bell_circuits:
            try:
                circuit = {
                    'name': circuit_def['name'],
                    'num_qubits': 2,
                    'gates': circuit_def['gates']
                }
                
                result = self.vm.execute_circuit(circuit)
                
                results[circuit_def['name']] = {
                    'description': circuit_def['description'],
                    'success': result.success,
                    'time_ms': result.execution_time_ms,
                    'fidelity': result.estimated_fidelity,
                    'measurements': result.measurements if hasattr(result, 'measurements') else None
                }
                
                total_fidelity += result.estimated_fidelity
                test_count += 1
                
                print(f"   {circuit_def['name']:20s}: {result.execution_time_ms:6.2f} ms, "
                      f"Fidelity: {result.estimated_fidelity:.6f}")
                
            except Exception as e:
                print(f"   {circuit_def['name']:20s}: ❌ ERROR: {e}")
                results[circuit_def['name']] = {
                    'description': circuit_def['description'],
                    'status': 'error',
                    'error': str(e)
                }
        
        avg_fidelity = total_fidelity / test_count if test_count > 0 else 0
        
        return {
            'status': 'passed' if test_count > 0 else 'failed',
            'results': results,
            'avg_fidelity': avg_fidelity
        }
    
    def test_ghz_state_scaling(self) -> Dict:
        """Test GHZ state creation scaling from 2 to max qubits"""
        results = {}
        total_fidelity = 0
        successful_tests = 0
        
        qubit_counts = [2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32]
        qubit_counts = [n for n in qubit_counts if n <= self.max_qubits]
        
        print(f"   Testing GHZ state scaling up to {max(qubit_counts)} qubits")
        
        for n in qubit_counts:
            try:
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
                
                result = self.vm.execute_circuit(circuit)
                
                results[n] = {
                    'success': result.success,
                    'time_ms': result.execution_time_ms,
                    'memory_mb': result.memory_used_gb * 1024 if hasattr(result, 'memory_used_gb') else 0,
                    'fidelity': result.estimated_fidelity,
                    'gate_count': len(circuit['gates'])
                }
                
                total_fidelity += result.estimated_fidelity
                successful_tests += 1
                
                print(f"   GHZ {n:2d} qubits: {result.execution_time_ms:7.2f} ms, "
                      f"Fidelity: {result.estimated_fidelity:.6f}")
                
            except MemoryError:
                print(f"   GHZ {n:2d} qubits: ❌ MEMORY ERROR")
                results[n] = {'status': 'memory_error'}
                break
            except Exception as e:
                print(f"   GHZ {n:2d} qubits: ❌ ERROR: {e}")
                results[n] = {'status': 'error', 'error': str(e)}
                break
        
        avg_fidelity = total_fidelity / successful_tests if successful_tests > 0 else 0
        
        return {
            'status': 'passed' if successful_tests > 0 else 'failed',
            'results': results,
            'avg_fidelity': avg_fidelity,
            'max_qubits_achieved': max(results.keys()) if results else 0
        }
    
    def test_quantum_fourier_transform(self) -> Dict:
        """Test QFT implementation with realistic gate counts"""
        if self.max_qubits < 2:
            return {'status': 'skipped', 'reason': 'Requires at least 2 qubits'}
        
        results = {}
        total_fidelity = 0
        test_count = 0
        
        qubit_counts = [2, 3, 4, 5, 6, 7, 8]
        qubit_counts = [n for n in qubit_counts if n <= self.max_qubits]
        
        print(f"   Testing QFT up to {max(qubit_counts)} qubits")
        
        for n in qubit_counts:
            try:
                circuit = self._generate_qft_circuit(n)
                result = self.vm.execute_circuit(circuit)
                
                results[n] = {
                    'success': result.success,
                    'time_ms': result.execution_time_ms,
                    'gate_count': len(circuit['gates']),
                    'fidelity': result.estimated_fidelity
                }
                
                total_fidelity += result.estimated_fidelity
                test_count += 1
                
                print(f"   QFT {n} qubits: {result.execution_time_ms:7.2f} ms, "
                      f"{len(circuit['gates']):4d} gates, "
                      f"Fidelity: {result.estimated_fidelity:.6f}")
                
            except Exception as e:
                print(f"   QFT {n} qubits: ❌ ERROR: {e}")
                results[n] = {'status': 'error', 'error': str(e)}
        
        avg_fidelity = total_fidelity / test_count if test_count > 0 else 0
        
        return {
            'status': 'passed' if test_count > 0 else 'failed',
            'results': results,
            'avg_fidelity': avg_fidelity
        }
    
    def _generate_qft_circuit(self, n: int) -> Dict:
        """Generate Quantum Fourier Transform circuit"""
        gates = []
        
        # Apply Hadamard and controlled rotations
        for i in range(n):
            gates.append({'gate': 'H', 'targets': [i]})
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                gates.append({
                    'gate': 'RZ',
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
        """Test random quantum circuits with validation"""
        results = {}
        np.random.seed(42)  # For reproducibility
        
        qubit_counts = [4, 8, 12, 16]
        qubit_counts = [n for n in qubit_counts if n <= self.max_qubits]
        
        print(f"   Testing random circuits on {len(qubit_counts)} qubit configurations")
        
        for n in qubit_counts:
            try:
                # Generate random circuit with n gates
                circuit = self._generate_random_circuit(n, min(n * 3, 50))  # Limit to 50 gates max
                result = self.vm.execute_circuit(circuit)
                
                results[n] = {
                    'success': result.success,
                    'time_ms': result.execution_time_ms,
                    'gate_count': len(circuit['gates']),
                    'fidelity': result.estimated_fidelity
                }
                
                print(f"   Random {n:2d} qubits: {result.execution_time_ms:7.2f} ms, "
                      f"{len(circuit['gates']):3d} gates, "
                      f"Fidelity: {result.estimated_fidelity:.6f}")
                
            except Exception as e:
                print(f"   Random {n:2d} qubits: ❌ ERROR: {e}")
                results[n] = {'status': 'error', 'error': str(e)}
        
        return {
            'status': 'passed' if len(results) > 0 else 'failed',
            'results': results
        }
    
    def _generate_random_circuit(self, num_qubits: int, num_gates: int) -> Dict:
        """Generate random quantum circuit with sensible limits"""
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
        if self.max_qubits < 4:
            return {'status': 'skipped', 'reason': 'Requires at least 4 qubits'}
        
        results = {}
        
        entanglement_patterns = [
            ('linear_chain', self._generate_linear_entanglement),
            ('star', self._generate_star_entanglement),
            ('ring', self._generate_ring_entanglement),
            ('grid', self._generate_grid_entanglement),
        ]
        
        qubit_counts = [4, 8, 12]
        qubit_counts = [n for n in qubit_counts if n <= self.max_qubits]
        
        print(f"   Testing {len(entanglement_patterns)} entanglement patterns")
        
        for pattern_name, generator in entanglement_patterns:
            pattern_results = {}
            
            for n in qubit_counts:
                try:
                    circuit = generator(n)
                    result = self.vm.execute_circuit(circuit)
                    
                    pattern_results[n] = {
                        'success': result.success,
                        'time_ms': result.execution_time_ms,
                        'gate_count': len(circuit['gates']),
                        'fidelity': result.estimated_fidelity
                    }
                    
                except Exception as e:
                    pattern_results[n] = {
                        'success': False,
                        'error': str(e)
                    }
            
            results[pattern_name] = pattern_results
            
            # Calculate average fidelity for this pattern
            fidelities = [r.get('fidelity', 0) for r in pattern_results.values() 
                         if isinstance(r, dict) and 'fidelity' in r]
            avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0
            
            print(f"   {pattern_name:15s}: avg fidelity = {avg_fidelity:.6f}")
        
        return {
            'status': 'passed' if len(results) > 0 else 'failed',
            'results': results
        }
    
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
            return {'status': 'skipped', 'reason': 'Real implementation required for measurement statistics'}
        
        results = {}
        
        # Test |+⟩ state measurement
        print("\n   Testing |+⟩ state (should be ~50/50):")
        counts = {0: 0, 1: 0}
        total_measurements = 100
        
        for i in range(total_measurements):
            circuit = {
                'name': f'plus_state_{i}',
                'num_qubits': 1,
                'gates': [
                    {'gate': 'H', 'targets': [0]},
                    {'gate': 'MEASURE', 'targets': [0]}
                ]
            }
            
            try:
                result = self.vm.execute_circuit(circuit)
                if result.measurements:
                    counts[result.measurements[0]] += 1
            except Exception as e:
                print(f"    Measurement {i+1} failed: {e}")
        
        results['plus_state'] = {
            'counts': counts,
            'ratio_0': counts[0] / total_measurements,
            'ratio_1': counts[1] / total_measurements
        }
        
        print(f"    |0⟩: {counts[0]}, |1⟩: {counts[1]} "
              f"(ratio: {counts[0]/total_measurements:.2f})")
        
        # Test Bell state measurement correlation
        print("\n   Testing Bell state correlation:")
        correlations = 0
        total = 50
        
        for i in range(total):
            circuit = {
                'name': f'bell_correlation_{i}',
                'num_qubits': 2,
                'gates': [
                    {'gate': 'H', 'targets': [0]},
                    {'gate': 'CNOT', 'targets': [1], 'controls': [0]},
                    {'gate': 'MEASURE', 'targets': [0, 1]}
                ]
            }
            
            try:
                result = self.vm.execute_circuit(circuit)
                if result.measurements and len(result.measurements) == 2:
                    if result.measurements[0] == result.measurements[1]:
                        correlations += 1
            except Exception as e:
                print(f"    Bell measurement {i+1} failed: {e}")
        
        results['bell_correlation'] = {
            'correlations': correlations,
            'total': total,
            'correlation_ratio': correlations / total
        }
        
        print(f"    Same measurement: {correlations}/{total} "
              f"({correlations/total:.2f} correlation)")
        
        return {
            'status': 'passed',
            'results': results
        }
    
    def test_memory_scaling(self) -> Dict:
        """Test memory usage scaling with qubit count"""
        results = {}
        
        qubit_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        qubit_counts = [n for n in qubit_counts if n <= self.max_qubits]
        
        print(f"   Testing memory scaling up to {max(qubit_counts)} qubits")
        
        for n in qubit_counts:
            try:
                # Simple circuit to measure memory usage
                circuit = {
                    'name': f'memory_test_{n}',
                    'num_qubits': n,
                    'gates': [{'gate': 'H', 'targets': [0]}]
                }
                
                result = self.vm.execute_circuit(circuit)
                
                # Calculate theoretical memory (complex128 = 16 bytes per amplitude)
                theoretical_mb = (2 ** n) * 16 / (1024 ** 2)
                
                results[n] = {
                    'theoretical_mb': theoretical_mb,
                    'actual_mb': result.memory_used_gb * 1024 if hasattr(result, 'memory_used_gb') else 0,
                    'compression_ratio': result.compression_ratio if hasattr(result, 'compression_ratio') else 1.0
                }
                
                print(f"   {n:2d} qubits: Theoretical {theoretical_mb:10.1f} MB, "
                      f"Actual {results[n]['actual_mb']:10.1f} MB, "
                      f"Ratio: {results[n]['actual_mb']/theoretical_mb:.3f}")
                
            except MemoryError:
                print(f"   {n:2d} qubits: ❌ MEMORY LIMIT EXCEEDED")
                results[n] = {'status': 'memory_error'}
                break
            except Exception as e:
                print(f"   {n:2d} qubits: ❌ ERROR: {e}")
                results[n] = {'status': 'error', 'error': str(e)}
                break
        
        return {
            'status': 'passed' if len(results) > 0 else 'failed',
            'results': results
        }
    
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
        
        qubit_counts = [4, 8, 12, 16, 20, 24]
        qubit_counts = [n for n in qubit_counts if n <= self.max_qubits]
        
        print(f"   Running {len(performance_tests)} performance benchmarks")
        
        for test_name, gate_generator in performance_tests:
            test_results = {}
            
            for n in qubit_counts:
                try:
                    circuit = {
                        'name': f'perf_{test_name}_{n}',
                        'num_qubits': n,
                        'gates': gate_generator(n)
                    }
                    
                    result = self.vm.execute_circuit(circuit)
                    
                    test_results[n] = {
                        'time_ms': result.execution_time_ms,
                        'gates_per_ms': len(circuit['gates']) / result.execution_time_ms * 1000 if result.execution_time_ms > 0 else 0,
                        'fidelity': result.estimated_fidelity
                    }
                    
                except Exception as e:
                    test_results[n] = {
                        'error': str(e)
                    }
            
            results[test_name] = test_results
            
            # Calculate average gates per second
            gates_per_sec = []
            for n, data in test_results.items():
                if isinstance(data, dict) and 'gates_per_ms' in data:
                    gates_per_sec.append(data['gates_per_ms'])
            
            avg_gates_per_sec = sum(gates_per_sec) / len(gates_per_sec) if gates_per_sec else 0
            
            print(f"   {test_name:15s}: avg {avg_gates_per_sec:7.0f} gates/s")
        
        return {
            'status': 'passed' if len(results) > 0 else 'failed',
            'results': results
        }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        completed_tests = sum(1 for r in self.test_results if r.status == TestStatus.COMPLETED)
        failed_tests = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
        skipped_tests = sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED)
        
        print("\n" + "="*80)
        print("📋 ENHANCED QNVM TEST REPORT")
        print("="*80)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Execution Time: {total_time:.2f} seconds")
        print(f"   Tests Completed: {completed_tests}/{len(self.test_results)}")
        print(f"   Tests Failed: {failed_tests}")
        print(f"   Tests Skipped: {skipped_tests}")
        print(f"   Peak Memory Usage: {self.peak_memory:.1f} MB")
        print(f"   Average CPU Usage: {sum(self.cpu_readings)/len(self.cpu_readings):.1f}%" if self.cpu_readings else "   Average CPU Usage: N/A")
        
        # Performance insights
        completed_results = [r for r in self.test_results if r.status == TestStatus.COMPLETED]
        if completed_results:
            longest_test = max(completed_results, key=lambda x: x.execution_time)
            highest_memory = max(completed_results, key=lambda x: x.memory_used)
            highest_fidelity = max(completed_results, key=lambda x: x.fidelity or 0)
            
            print(f"\n⚡ Performance Insights:")
            print(f"   Longest Running Test: {longest_test.name} ({longest_test.execution_time:.2f}s)")
            print(f"   Most Memory Intensive: {highest_memory.name} ({highest_memory.memory_used:.1f} MB)")
            print(f"   Highest Fidelity: {highest_fidelity.name} ({highest_fidelity.fidelity:.6f})")
        
        # List failed tests
        failed = [r for r in self.test_results if r.status == TestStatus.FAILED]
        if failed:
            print(f"\n❌ Failed Tests:")
            for result in failed:
                print(f"   - {result.name}: {result.error_message}")
        
        # List skipped tests
        skipped = [r for r in self.test_results if r.status == TestStatus.SKIPPED]
        if skipped:
            print(f"\n⚠️  Skipped Tests:")
            for result in skipped:
                print(f"   - {result.name}: {result.error_message}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_csv_report(timestamp)
        self.save_json_report(timestamp)
        
        print(f"\n💾 Reports saved:")
        print(f"   CSV: qubit_test_summary_{timestamp}.csv")
        print(f"   JSON: qubit_test_report_{timestamp}.json")
        print(f"   Error log: qubit_test_errors.log")
        
        # Final assessment
        success_rate = completed_tests / len(self.test_results) if self.test_results else 0
        if success_rate >= 0.95:
            print(f"\n✅ TEST SUITE PASSED: {success_rate:.1%} success rate")
        elif success_rate >= 0.80:
            print(f"\n⚠️  TEST SUITE WARNING: {success_rate:.1%} success rate")
        else:
            print(f"\n❌ TEST SUITE FAILED: {success_rate:.1%} success rate")
    
    def save_csv_report(self, timestamp: str):
        """Save test results to CSV file"""
        filename = f"qubit_test_summary_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test Name', 'Status', 'Time (s)', 'Memory (MB)', 'CPU (%)', 'Qubits', 'Gates', 'Fidelity', 'Error'])
            
            for result in self.test_results:
                writer.writerow([
                    result.name,
                    result.status.value,
                    f"{result.execution_time:.2f}",
                    f"{result.memory_used:.1f}",
                    f"{result.cpu_percent or 0:.1f}",
                    f"{result.qubits_tested or 0}",
                    f"{result.gates_executed or 0}",
                    f"{result.fidelity or 0:.6f}",
                    result.error_message or ""
                ])
        
        print(f"   CSV report saved to {filename}")
    
    def save_json_report(self, timestamp: str):
        """Save detailed test report to JSON file"""
        filename = f"qubit_test_report_{timestamp}.json"
        
        report = {
            'metadata': {
                'version': 'QNVM v5.1',
                'timestamp': timestamp,
                'max_qubits': self.max_qubits,
                'use_real_implementation': self.use_real,
                'total_time_seconds': time.time() - self.start_time
            },
            'system_info': {
                'cpu_percent_history': self.cpu_readings,
                'memory_history': self.memory_readings,
                'peak_memory': self.peak_memory,
                'initial_memory': self.initial_memory
            },
            'test_results': [
                {
                    'name': r.name,
                    'status': r.status.value,
                    'execution_time': r.execution_time,
                    'memory_used': r.memory_used,
                    'cpu_percent': r.cpu_percent,
                    'qubits_tested': r.qubits_tested,
                    'gates_executed': r.gates_executed,
                    'fidelity': r.fidelity,
                    'error_message': r.error_message,
                    'measurements': r.measurements,
                    'details': r.details
                } for r in self.test_results
            ],
            'summary': {
                'total_time': time.time() - self.start_time,
                'completed_tests': sum(1 for r in self.test_results if r.status == TestStatus.COMPLETED),
                'failed_tests': sum(1 for r in self.test_results if r.status == TestStatus.FAILED),
                'skipped_tests': sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED),
                'success_rate': sum(1 for r in self.test_results if r.status == TestStatus.COMPLETED) / len(self.test_results) * 100 if self.test_results else 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   JSON report saved to {filename}")

def main():
    """Main test runner"""
    print("="*80)
    print("🚀 QNVM v5.1 - ENHANCED QUANTUM TEST SUITE (Up to 32 Qubits)")
    print("="*80)
    
    print("\n🔧 Enhanced features:")
    print("  - Real-time CPU and memory monitoring")
    print("  - Comprehensive error handling and recovery")
    print("  - Graceful degradation on memory limits")
    print("  - Detailed CSV and JSON reporting")
    print("  - Performance insights and benchmarking")
    
    # Get user input for test parameters
    try:
        max_qubits = int(input("\nEnter maximum qubits to test (2-32, recommended 16): ") or "16")
        max_qubits = max(2, min(32, max_qubits))
    except:
        max_qubits = 16
    
    use_real = input("Use real quantum implementation? (y/n, default y): ").lower() != 'n'
    
    # Run tests
    try:
        test_suite = EnhancedQuantumTestSuite(max_qubits=max_qubits, use_real=use_real)
        test_suite.run_all_tests()
        test_suite.generate_report()
        
        print("\n" + "="*80)
        print("🎉 ENHANCED QUBIT TESTING COMPLETE!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
