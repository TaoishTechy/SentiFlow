"""
Quantum Neuro Virtual Machine - Core Implementation
"""

import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import warnings

# Import from external modules
from external import (
    TensorNetwork, MPS, ContractOptimizer,
    QuantumMemoryManager,
    QuantumProcessor, VirtualQubit,
    QuantumErrorCorrection, SurfaceCode, NeuralMWPMDecoder,
    QuantumGate, GateCompiler, LogicalOperationCompiler,
    SparseQuantumState, CompressedState,
    StateCompressor,
    QuantumStateValidator, GroundTruthVerifier,
    QiskitBackend, CirqBackend, BackendManager
)

from .config import QNVMConfig, BackendType, CompressionMethod

@dataclass
class CircuitResult:
    """Result of circuit execution"""
    success: bool
    execution_time_ms: float
    memory_used_gb: float
    estimated_fidelity: float
    state_representation: str
    compression_ratio: float
    validation_passed: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return asdict(self)

class QNVM:
    """Quantum Neuro Virtual Machine - Main Class"""
    
    def __init__(self, config: Optional[QNVMConfig] = None):
        """Initialize QNVM with configuration"""
        self.config = config or QNVMConfig()
        
        # Validate configuration
        if errors := self.config.validate():
            for error in errors:
                warnings.warn(f"Configuration warning: {error}")
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize modules
        self._initialize_modules()
        
        # State tracking
        self.current_state = None
        self.circuit_history = []
        self.performance_stats = {
            'total_gates': 0,
            'total_time_ns': 0,
            'total_circuits': 0,
            'memory_usage_history': [],
            'fidelity_history': [],
            'execution_times': []
        }
        
        # Cache for compiled circuits
        self.circuit_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.logger.info(f"QNVM v5.1 initialized with config: {self.config.to_dict()}")
        self.logger.info(f"Memory requirements: {self.config.get_memory_requirements()}")
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('qnvm.log')
            ]
        )
    
    def _initialize_modules(self):
        """Initialize all QNVM modules"""
        try:
            # Memory management
            self.memory_manager = QuantumMemoryManager(
                max_memory_gb=self.config.max_memory_gb
            )
            
            # Quantum processor
            self.processor = QuantumProcessor(
                num_qubits=self.config.max_qubits
            )
            
            # Tensor network backend
            self.tensor_network = TensorNetwork(
                num_qubits=self.config.max_qubits
            )
            
            # Error correction
            if self.config.error_correction:
                self.error_correction = QuantumErrorCorrection(
                    code_type=self.config.code_type,
                    distance=self.config.code_distance
                )
            else:
                self.error_correction = None
            
            # Gate compiler
            self.gate_compiler = GateCompiler()
            
            # State management
            self.state_compressor = CompressedState()
            self.state_validator = QuantumStateValidator()
            
            # External backend integration
            if self.config.backend == BackendType.QISKIT:
                self.backend = QiskitBackend(
                    **self.config.backend_options.get('qiskit', {})
                )
            elif self.config.backend == BackendType.CIRQ:
                self.backend = CirqBackend(
                    **self.config.backend_options.get('cirq', {})
                )
            else:
                self.backend = None  # Use internal tensor network
            
            # Backend manager for runtime switching
            self.backend_manager = BackendManager()
            
            self.logger.info("All modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize modules: {e}")
            raise
    
    def execute_circuit(self, circuit: Dict) -> CircuitResult:
        """
        Execute a quantum circuit
        
        Args:
            circuit: Dictionary containing circuit definition
                   {
                     'name': str,
                     'num_qubits': int,
                     'gates': List[Dict],
                     'type': str (optional),
                     'metadata': Dict (optional)
                   }
        
        Returns:
            CircuitResult object with execution results
        """
        start_time = time.time()
        
        try:
            # Extract circuit information
            circuit_name = circuit.get('name', f'circuit_{len(self.circuit_history)}')
            num_qubits = circuit.get('num_qubits', self.config.max_qubits)
            gates = circuit.get('gates', [])
            circuit_type = circuit.get('type', 'generic')
            
            self.logger.info(f"Executing circuit: {circuit_name} ({num_qubits} qubits, {len(gates)} gates)")
            
            # Validate circuit
            if num_qubits > self.config.max_qubits:
                raise ValueError(
                    f"Circuit requires {num_qubits} qubits, "
                    f"but QNVM is configured for {self.config.max_qubits} qubits"
                )
            
            # Check cache
            circuit_hash = self._hash_circuit(circuit)
            if self.config.enable_caching and circuit_hash in self.circuit_cache:
                self.cache_hits += 1
                cached_result = self.circuit_cache[circuit_hash]
                self.logger.debug(f"Cache hit for circuit: {circuit_name}")
                return cached_result
            
            self.cache_misses += 1
            
            # Allocate memory for quantum state
            memory_info = self.memory_manager.allocate_state(
                num_qubits=num_qubits,
                state_id=circuit_name,
                sparse_threshold=self.config.sparse_threshold
            )
            
            # Initialize quantum state
            state = self._initialize_state(num_qubits, memory_info['representation'])
            
            # Execute gates
            gate_errors = []
            for i, gate_info in enumerate(gates):
                gate_result = self._execute_gate(
                    state=state,
                    gate_info=gate_info,
                    gate_index=i
                )
                
                if isinstance(gate_result, tuple):
                    state, error_prob = gate_result
                    gate_errors.append(error_prob)
                else:
                    state = gate_result
                
                self.performance_stats['total_gates'] += 1
            
            # Apply error correction if enabled
            if self.error_correction:
                state = self._apply_error_correction(
                    state=state,
                    gate_errors=gate_errors
                )
            
            # Compress state if enabled
            if self.config.compression_enabled:
                state = self.state_compressor.compress(
                    state=state,
                    ratio=self.config.compression_ratio,
                    method=self.config.compression_method.value
                )
            
            # Validate state
            validation_result = None
            if self.config.validation_enabled:
                validation_result = self.state_validator.validate_state(state)
                if not validation_result['valid']:
                    self.logger.warning(f"State validation failed: {validation_result['errors']}")
            
            # Calculate metrics
            execution_time = time.time() - start_time
            memory_stats = self.memory_manager.get_memory_stats()
            
            # Calculate fidelity
            fidelity = self._calculate_fidelity(gate_errors, len(gates))
            
            # Create result
            result = CircuitResult(
                success=True,
                execution_time_ms=execution_time * 1000,
                memory_used_gb=memory_stats['quantum_memory_usage_gb'],
                estimated_fidelity=fidelity,
                state_representation=memory_info['representation'],
                compression_ratio=memory_info.get('compression_ratio', 1.0),
                validation_passed=validation_result.get('valid') if validation_result else None,
                metadata={
                    'num_qubits': num_qubits,
                    'num_gates': len(gates),
                    'circuit_type': circuit_type,
                    'cache_hit': False
                }
            )
            
            # Update performance stats
            self._update_performance_stats(result, execution_time)
            
            # Cache result
            if self.config.enable_caching:
                self._cache_result(circuit_hash, result)
            
            # Store circuit in history
            self.circuit_history.append({
                'name': circuit_name,
                'circuit': circuit,
                'result': result,
                'timestamp': time.time(),
                'state_hash': self._hash_state(state) if num_qubits <= 16 else None
            })
            
            self.logger.info(
                f"Circuit execution completed: {circuit_name} "
                f"(time: {result.execution_time_ms:.2f}ms, "
                f"fidelity: {result.estimated_fidelity:.6f})"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Circuit execution failed: {e}")
            
            return CircuitResult(
                success=False,
                execution_time_ms=execution_time * 1000,
                memory_used_gb=0.0,
                estimated_fidelity=0.0,
                state_representation="error",
                compression_ratio=1.0,
                error_message=str(e),
                metadata={'error': str(e)}
            )
    
    def _initialize_state(self, num_qubits: int, representation: str) -> Any:
        """Initialize quantum state based on representation type"""
        if representation == 'tensor_mps':
            mps = MPS(num_qubits)
            return mps.to_statevector()
        elif representation == 'sparse':
            sparse_state = SparseQuantumState(num_qubits, self.config.sparse_threshold)
            # Initialize to |0⟩^n
            dense_state = np.zeros(2 ** num_qubits, dtype=np.complex128)
            dense_state[0] = 1.0
            return sparse_state.from_dense(dense_state)
        else:  # dense
            state = np.zeros(2 ** num_qubits, dtype=np.complex128)
            state[0] = 1.0
            return state
    
    def _execute_gate(self, state: Any, gate_info: Dict, gate_index: int) -> Any:
        """Execute a single quantum gate"""
        gate_name = gate_info.get('gate', '')
        targets = gate_info.get('targets', [])
        controls = gate_info.get('controls', [])
        params = gate_info.get('params', {})
        
        # Use external backend if configured
        if self.backend:
            return self.backend.apply_gate(state, gate_name, targets, controls, params)
        
        # Otherwise use internal implementation
        return self._execute_gate_internal(state, gate_name, targets, controls, params)
    
    def _execute_gate_internal(self, state: Any, gate_name: str,
                             targets: List[int], controls: List[int],
                             params: Dict) -> Tuple[Any, float]:
        """Execute gate using internal implementation"""
        # Placeholder - would use actual gate implementations
        error_prob = 0.0
        
        if self.processor:
            error_prob = self.processor.execute_gate(gate_name, targets, controls)
        
        # For now, return state unchanged with error probability
        return state, error_prob
    
    def _apply_error_correction(self, state: Any, gate_errors: List[float]) -> Any:
        """Apply error correction to state"""
        if not self.error_correction or not gate_errors:
            return state
        
        avg_error = np.mean(gate_errors) if gate_errors else 0.001
        cycle = len(self.circuit_history)
        
        # Convert state to numpy array for error correction
        if hasattr(state, 'to_dense'):
            state_array = state.to_dense()
        else:
            state_array = state
        
        corrected_state = self.error_correction.run_correction_cycle(
            state=state_array,
            cycle=cycle,
            noise_rate=avg_error
        )
        
        # Convert back to original representation
        if hasattr(state, 'from_dense'):
            return state.from_dense(corrected_state)
        
        return corrected_state
    
    def _calculate_fidelity(self, gate_errors: List[float], num_gates: int) -> float:
        """Calculate overall circuit fidelity"""
        if not gate_errors:
            return 1.0
        
        avg_error = np.mean(gate_errors)
        
        # Simplified fidelity model
        if self.config.error_correction:
            # With error correction, errors are suppressed
            logical_error = avg_error ** ((self.config.code_distance + 1) // 2)
            fidelity = np.exp(-logical_error * num_gates)
        else:
            # Without error correction
            fidelity = np.exp(-avg_error * num_gates)
        
        return min(1.0, max(0.0, fidelity))
    
    def _hash_circuit(self, circuit: Dict) -> str:
        """Create hash for circuit caching"""
        import hashlib
        
        # Create deterministic string representation
        circuit_str = json.dumps(circuit, sort_keys=True)
        return hashlib.md5(circuit_str.encode()).hexdigest()
    
    def _hash_state(self, state: Any) -> str:
        """Create hash for quantum state"""
        import hashlib
        
        if hasattr(state, 'to_dense'):
            state_array = state.to_dense()
        else:
            state_array = state
        
        # Convert to bytes and hash
        state_bytes = state_array.tobytes()
        return hashlib.md5(state_bytes).hexdigest()
    
    def _update_performance_stats(self, result: CircuitResult, execution_time: float):
        """Update performance statistics"""
        self.performance_stats['total_time_ns'] += execution_time * 1e9
        self.performance_stats['total_circuits'] += 1
        self.performance_stats['execution_times'].append(execution_time)
        self.performance_stats['memory_usage_history'].append(result.memory_used_gb)
        self.performance_stats['fidelity_history'].append(result.estimated_fidelity)
        
        # Keep only recent history
        if len(self.performance_stats['execution_times']) > 1000:
            self.performance_stats['execution_times'] = self.performance_stats['execution_times'][-1000:]
            self.performance_stats['memory_usage_history'] = self.performance_stats['memory_usage_history'][-1000:]
            self.performance_stats['fidelity_history'] = self.performance_stats['fidelity_history'][-1000:]
    
    def _cache_result(self, circuit_hash: str, result: CircuitResult):
        """Cache circuit result"""
        # Check cache size
        if len(self.circuit_cache) >= self.config.cache_size_mb * 1024 // 8:  # Rough estimate
            # Remove oldest entry
            oldest_key = next(iter(self.circuit_cache))
            del self.circuit_cache[oldest_key]
        
        self.circuit_cache[circuit_hash] = result
    
    def get_statistics(self) -> Dict:
        """Get comprehensive QNVM statistics"""
        if not self.performance_stats['execution_times']:
            avg_time = 0
            std_time = 0
        else:
            times = self.performance_stats['execution_times']
            avg_time = np.mean(times) * 1000  # ms
            std_time = np.std(times) * 1000   # ms
        
        return {
            'config': self.config.to_dict(),
            'performance': {
                'total_circuits': self.performance_stats['total_circuits'],
                'total_gates': self.performance_stats['total_gates'],
                'total_time_ms': self.performance_stats['total_time_ns'] / 1e6,
                'avg_execution_time_ms': avg_time,
                'std_execution_time_ms': std_time,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                'avg_fidelity': np.mean(self.performance_stats['fidelity_history']) if self.performance_stats['fidelity_history'] else 0,
                'avg_memory_gb': np.mean(self.performance_stats['memory_usage_history']) if self.performance_stats['memory_usage_history'] else 0
            },
            'memory': self.memory_manager.get_memory_stats() if hasattr(self, 'memory_manager') else {},
            'processor': {
                'num_qubits': self.processor.num_qubits if hasattr(self, 'processor') else 0,
                'fidelity': self.processor.get_processor_fidelity() if hasattr(self, 'processor') else 0.0,
            } if hasattr(self, 'processor') else None,
            'error_correction': self.error_correction.get_statistics() if hasattr(self, 'error_correction') else None,
            'circuit_history_count': len(self.circuit_history),
            'backend': self.config.backend.value
        }
    
    def save_state(self, filename: str):
        """Save current QNVM state to file"""
        state_data = {
            'config': self.config.to_dict(),
            'performance': self.performance_stats,
            'circuit_history': [ch['name'] for ch in self.circuit_history[-100:]],  # Save only names
            'cache_stats': {
                'size': len(self.circuit_cache),
                'hits': self.cache_hits,
                'misses': self.cache_misses
            },
            'timestamp': time.time(),
            'version': '5.1.0'
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        self.logger.info(f"State saved to {filename}")
    
    def reset(self):
        """Reset QNVM to initial state"""
        self.current_state = None
        self.circuit_history = []
        self.performance_stats = {
            'total_gates': 0,
            'total_time_ns': 0,
            'total_circuits': 0,
            'memory_usage_history': [],
            'fidelity_history': [],
            'execution_times': []
        }
        self.circuit_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        if hasattr(self, 'memory_manager'):
            self.memory_manager.clear_all()
        
        self.logger.info("QNVM reset to initial state")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        if exc_type:
            self.logger.error(f"Exception in QNVM context: {exc_val}")
        
        # Cleanup resources
        if hasattr(self, 'memory_manager'):
            self.memory_manager.clear_all()
        
        self.logger.info("QNVM context exited")
