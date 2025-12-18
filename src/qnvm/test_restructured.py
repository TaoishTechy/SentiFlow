#!/usr/bin/env python3
"""
Test the restructured QNVM modules
"""

import sys
sys.path.insert(0, './src/external')

from qnvm import QNVM, QNVMConfig

def test_basic_functionality():
    """Test basic QNVM functionality"""
    print("Testing restructured QNVM...")
    
    # Create configuration
    config = QNVMConfig(
        max_qubits=16,
        max_memory_gb=8.0,
        backend='tensor_network',
        error_correction=True,
        code_distance=3,
        compression_enabled=True,
        validation_enabled=True
    )
    
    # Initialize QNVM
    vm = QNVM(config)
    
    # Test GHZ circuit
    circuit = {
        'num_qubits': 4,
        'type': 'ghz',
        'gates': [
            {'gate': 'H', 'targets': [0]},
            {'gate': 'CNOT', 'targets': [1], 'controls': [0]},
            {'gate': 'CNOT', 'targets': [2], 'controls': [0]},
            {'gate': 'CNOT', 'targets': [3], 'controls': [0]}
        ]
    }
    
    # Run simulation
    result = vm.simulate_32q_circuit(circuit)
    
    print(f"Simulation result: {result['success']}")
    print(f"Execution time: {result['execution_time_ms']:.2f} ms")
    print(f"Memory used: {result['memory_used_gb']:.4f} GB")
    print(f"Fidelity: {result['estimated_fidelity']:.6f}")
    print(f"Compression ratio: {result['compression_ratio']:.4f}")
    
    # Get diagnostics
    diag = vm.get_diagnostics()
    print(f"\nDiagnostics:")
    print(f"Total gates executed: {diag['performance']['total_gates']}")
    print(f"Total time: {diag['performance']['total_time_ns'] / 1e9:.2f} s")
    
    return result['success']

def test_module_imports():
    """Test that all modules can be imported"""
    modules = [
        'tensor_network',
        'quantum_memory', 
        'quantum_processor',
        'error_correction',
        'quantum_operations',
        'sparse_quantum_state'
    ]
    
    for module_name in modules:
        try:
            __import__(f'src.external.{module_name}')
            print(f"✓ {module_name} imports successfully")
        except ImportError as e:
            print(f"✗ {module_name} import failed: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("QuantumNeuroVM v5.1 Restructured Module Test")
    print("=" * 60)
    
    # Test module imports
    if not test_module_imports():
        print("\nModule import test failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Testing QNVM functionality...")
    print("=" * 60)
    
    # Test QNVM functionality
    if test_basic_functionality():
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Tests failed!")
        print("=" * 60)
        sys.exit(1)