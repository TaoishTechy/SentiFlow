# Using quantum_core_engine for optimized gate operations
class QuantumCoreAdapter:
    """Adapter for quantum_core_engine"""
    
    def __init__(self):
        if self._modules["quantum_core_engine"]["available"]:
            self.engine = self._modules["quantum_core_engine"]["module"]
            self._use_native_gates = True
        else:
            self._use_native_gates = False
            self._initialize_fallback_gates()
    
    def apply_quantum_gate(self, gate_name: str, targets: List[int], params: Dict = None):
        """Use quantum_core_engine for gate application"""
        if self._use_native_gates:
            # Use optimized gate from quantum_core_engine
            return self.engine.apply_gate(
                gate_type=gate_name,
                qubits=targets,
                parameters=params
            )
        else:
            # Fallback to our implementation
            return self._apply_fallback_gate(gate_name, targets, params)