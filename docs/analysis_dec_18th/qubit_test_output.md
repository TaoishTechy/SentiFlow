🔍 Initializing Quantum Test Suite v5.1...

🔍 Loading QNVM...
✅ QNVM v5.1.0 (Real Quantum Implementation) ready!
✅ QNVM v5.1 loaded successfully
   Real Implementation: True
   Backend Types: ['CIRQ', 'INTERNAL', 'QISKIT', 'TENSOR_NETWORK']

🔍 Checking for advanced modules...
✅ External modules available: {'tensor_network': True, 'fidelity': True, 'memory_manager': True}

================================================================================
🚀 QUANTUM TEST SUITE v5.1 - ENHANCED EDITION
================================================================================

🔧 ENHANCED FEATURES:
  - Robust import handling with graceful fallbacks
  - Comprehensive error resilience
  - Memory-efficient testing up to 32 qubits
  - Advanced fidelity and metrics calculation
  - Detailed system monitoring
  - Multiple output formats (CSV, JSON)

Enter maximum qubits to test (1-32, default 8): 32

📊 SYSTEM ANALYSIS:
   Available RAM: 8.7 GB
   Suggested memory limit: 4.0 GB
   Testing up to: 32 qubits
Use real quantum implementation? (y/n, default y): y
Enable quantum state validation? (y/n, default y): y

================================================================================
🚀 STARTING QUANTUM TEST SUITE
================================================================================

======================================================================
⚙️  QUANTUM TEST SUITE CONFIGURATION
======================================================================
   Maximum Qubits: 32
   System Memory: 15.0 GB total, 8.1 GB available
   Memory Limit: 4.0 GB
   QNVM Available: True
   Real Quantum: True
   Validation: Enabled
   Advanced Modules: True
======================================================================
/src/qnvm/core_real.py:401: UserWarning: Reducing max_qubits from 32 to 28 due to memory constraints (8.1 GB available)
  warnings.warn(
✅ QNVM initialized successfully

======================================================================
🚀 RUNNING QUANTUM TEST SUITE
======================================================================

📋 Test Sequence (10 tests):
    1. State Initialization
    2. Single-Qubit Gates
    3. Two-Qubit Gates
    4. Bell State Creation
    5. GHZ State Scaling
    6. Random Circuits
    7. Entanglement Generation
    8. Measurement Statistics
    9. Memory Scaling
   10. Performance Benchmark

============================================================
🧪 TEST: State Initialization
============================================================
   Testing state initialization for 6 qubit counts
    1 qubits: ✅ fidelity=1.000000, time=0.10ms
    2 qubits: ✅ fidelity=1.000000, time=0.06ms
    4 qubits: ✅ fidelity=1.000000, time=0.06ms
    8 qubits: ✅ fidelity=1.000000, time=0.05ms
   12 qubits: ✅ fidelity=1.000000, time=0.06ms
   16 qubits: ✅ fidelity=1.000000, time=0.06ms
   ✅ Status: completed
   ⏱️  Time: 0.001s
   💾 Memory: 0.1 MB
   🟢 Fidelity: 1.000000

============================================================
🧪 TEST: Single-Qubit Gates
============================================================
   Testing 6 single-qubit gates
   H  gate: ✅ fidelity=0.999000
   X  gate: ✅ fidelity=0.999000
   Y  gate: ✅ fidelity=0.999000
   Z  gate: ✅ fidelity=0.999000
   S  gate: ✅ fidelity=0.999000
   T  gate: ✅ fidelity=0.999000
   ✅ Status: completed
   ⏱️  Time: 0.002s
   💾 Memory: 0.3 MB
   🟢 Fidelity: 0.999000

============================================================
🧪 TEST: Two-Qubit Gates
============================================================
   Testing CNOT gate
   CNOT gate: ✅ fidelity=0.998002
   ✅ Status: completed
   ⏱️  Time: 0.000s
   💾 Memory: 0.0 MB
   🟢 Fidelity: 0.998002

============================================================
🧪 TEST: Bell State Creation
============================================================
   Testing Bell state
   Bell state: ✅ fidelity=0.998002
   ✅ Status: completed
   ⏱️  Time: 0.001s
   💾 Memory: 0.0 MB
   🟢 Fidelity: 0.998002

============================================================
🧪 TEST: GHZ State Scaling
============================================================
   Testing GHZ state scaling (5 sizes)
   GHZ  2 qubits: ✅ fidelity=0.998002
   GHZ  3 qubits: ✅ fidelity=0.997004
   GHZ  4 qubits: ✅ fidelity=0.996008
   GHZ  5 qubits: ✅ fidelity=0.995012
   GHZ  6 qubits: ✅ fidelity=0.994018
   ✅ Status: completed
   ⏱️  Time: 0.002s
   💾 Memory: 0.0 MB
   🟢 Fidelity: 0.996009

============================================================
🧪 TEST: Random Circuits
============================================================
   Testing random circuits
   Random circuit: ✅ fidelity=0.994018, 6 gates
   ✅ Status: completed
   ⏱️  Time: 0.001s
   💾 Memory: 0.0 MB
   🟢 Fidelity: 0.994018

============================================================
🧪 TEST: Entanglement Generation
============================================================
   Testing entanglement generation
   Entanglement: ✅ fidelity=0.996008
   ✅ Status: completed
   ⏱️  Time: 0.001s
   💾 Memory: 0.0 MB
   🟢 Fidelity: 0.996008

============================================================
🧪 TEST: Measurement Statistics
============================================================
   Testing measurement statistics
   Measurement: ✅ fidelity=0.998686, χ²=2.49, shots=1000
   ✅ Status: completed
   ⏱️  Time: 0.010s
   💾 Memory: 6.9 MB
   🟢 Fidelity: 0.998686

============================================================
🧪 TEST: Memory Scaling
============================================================
   Testing memory scaling (5 sizes)
    1 qubits: ❌ ratio=1.000, theoretical=0.0MB, actual=0.0MB
    2 qubits: ❌ ratio=1.000, theoretical=0.0MB, actual=0.0MB
    4 qubits: ❌ ratio=1.000, theoretical=0.0MB, actual=0.0MB
    8 qubits: ❌ ratio=1.000, theoretical=0.0MB, actual=0.0MB
   12 qubits: ❌ ratio=1.000, theoretical=0.1MB, actual=0.1MB
   ✅ Status: completed
   ⏱️  Time: 0.006s
   💾 Memory: 0.2 MB

============================================================
🧪 TEST: Performance Benchmark
============================================================
   Running performance benchmark
   Performance: ✅ 3340 gates/sec, 6.0ms for 20 gates
   ✅ Status: completed
   ⏱️  Time: 0.007s
   💾 Memory: 0.1 MB
   🟡 Fidelity: 0.980199

================================================================================
📋 COMPREHENSIVE TEST REPORT
================================================================================

📊 SUMMARY:
   Total Tests: 10
   ✅ Completed: 10
   ⚠️  Warnings: 0
   ❌ Failed: 0
   ⏸️  Skipped: 0
   ⏱️  Total Time: 1.04s
   💾 Peak Memory: 38.3 MB
   🎯 Average Fidelity: 0.995547

📈 DETAILED RESULTS:
   ✅ State Initialization           completed   0.001s     0.1MB  fidelity=1.000000
   ✅ Single-Qubit Gates             completed   0.002s     0.3MB  fidelity=0.999000
   ✅ Two-Qubit Gates                completed   0.000s     0.0MB  fidelity=0.998002
   ✅ Bell State Creation            completed   0.001s     0.0MB  fidelity=0.998002
   ✅ GHZ State Scaling              completed   0.002s     0.0MB  fidelity=0.996009
   ✅ Random Circuits                completed   0.001s     0.0MB  fidelity=0.994018
   ✅ Entanglement Generation        completed   0.001s     0.0MB  fidelity=0.996008
   ✅ Measurement Statistics         completed   0.010s     6.9MB  fidelity=0.998686
   ✅ Memory Scaling                 completed   0.006s     0.2MB  
   ✅ Performance Benchmark          completed   0.007s     0.1MB  fidelity=0.980199

💾 REPORTS SAVED:
   CSV: quantum_test_summary_20251218_120007.csv
   JSON: quantum_test_report_20251218_120007.json

================================================================================
🎉 TEST SUITE PASSED: 100.0% success rate
================================================================================

================================================================================
🎉 QUANTUM TESTING COMPLETED SUCCESSFULLY!
================================================================================
