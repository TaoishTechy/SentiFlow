# Using sentiflow as the primary circuit framework
class SentiFlowAdapter:
    """Adapter for sentiflow quantum framework"""
    
    def __init__(self, integrator: AdvancedQuantumModuleIntegrator):
        self.integrator = integrator
        
        if integrator.modules.get("sentiflow", {}).get("available"):
            self.sentiflow = integrator.modules["sentiflow"]["module"]
            self._has_sentiflow = True
            self._initialize_sentiflow_backend()
        else:
            self._has_sentiflow = False
            self.backend = "numpy"
    
    def simulate_circuit(self, circuit_spec: Dict) -> Dict:
        """Simulate quantum circuit using sentiflow if available"""
        if self._has_sentiflow:
            # Convert to sentiflow circuit
            sf_circuit = self._to_sentiflow_circuit(circuit_spec)
            
            # Run simulation
            result = self.sentiflow.simulate(
                circuit=sf_circuit,
                shots=circuit_spec.get("shots", 1000),
                backend=self.backend
            )
            
            return {
                "framework": "sentiflow",
                "result": result,
                "circuit_info": sf_circuit.info()
            }
        else:
            # Use internal simulation
            return self._simulate_internal(circuit_spec)
