# Integrating qybrik for modular circuit construction
class QuantumCircuitBuilder:
    """Build circuits using qybrik building blocks"""
    
    def __init__(self, integrator: AdvancedQuantumModuleIntegrator):
        self.integrator = integrator
        self.qybrik = None
        
        if integrator.modules.get("qybrik", {}).get("available"):
            self.qybrik = integrator.modules["qybrik"]["module"]
            self._circuit_blocks = self._load_qybrik_blocks()
    
    def build_controlled_gate(self, control_qubits: List[int], 
                             target_gate: str, **kwargs):
        """Build controlled gates using qybrik"""
        if self.qybrik:
            # Use qybrik's controlled gate construction
            return self.qybrik.controlled_gate(
                controls=control_qubits,
                gate=target_gate,
                **kwargs
            )
        else:
            # Build manually using tensor products
            return self._build_controlled_gate_manual(control_qubits, target_gate)