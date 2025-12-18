import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class QubitState(Enum):
    """Qubit state enumeration"""
    GROUND = 0
    EXCITED = 1
    SUPERPOSITION = 2
    ENTANGLED = 3

@dataclass
class QubitProperties:
    """Physical properties of a qubit"""
    t1_time: float  # Relaxation time (ns)
    t2_time: float  # Dephasing time (ns)
    frequency: float  # Resonance frequency (GHz)
    anharmonicity: float  # Anharmonicity (MHz)
    readout_fidelity: float  # Measurement fidelity
    gate_fidelity: Dict[str, float]  # Gate fidelities

class VirtualQubit:
    """Virtual representation of a physical qubit"""
    def __init__(self, qubit_id: int, properties: Optional[QubitProperties] = None):
        self.qubit_id = qubit_id
        self.properties = properties or QubitProperties(
            t1_time=10000,  # 10 µs
            t2_time=5000,   # 5 µs
            frequency=5.0,  # 5 GHz
            anharmonicity=-300,  # -300 MHz
            readout_fidelity=0.99,
            gate_fidelity={'single': 0.999, 'two_qubit': 0.995}
        )
        self.state = QubitState.GROUND
        self.last_operation_time = 0
        self.error_history = []
        self.calibration_data = {}
    
    def apply_gate(self, gate_type: str, time_ns: float) -> float:
        """Apply gate with error probability"""
        if gate_type == 'single':
            fidelity = self.properties.gate_fidelity['single']
        elif gate_type == 'two_qubit':
            fidelity = self.properties.gate_fidelity['two_qubit']
        else:
            fidelity = 0.99  # Default
        
        # Calculate error probability based on time
        t1_error = np.exp(-time_ns / self.properties.t1_time)
        t2_error = np.exp(-time_ns / self.properties.t2_time)
        
        total_fidelity = fidelity * t1_error * t2_error
        error_prob = 1 - total_fidelity
        
        self.error_history.append({
            'time': time_ns,
            'gate_type': gate_type,
            'error_prob': error_prob,
            'fidelity': total_fidelity
        })
        
        self.last_operation_time += time_ns
        return error_prob
    
    def measure(self) -> Tuple[int, float]:
        """Measure qubit with readout fidelity"""
        # Simulate measurement with error
        true_value = 0 if self.state == QubitState.GROUND else 1
        if np.random.random() > self.properties.readout_fidelity:
            measured = 1 - true_value  # Error
        else:
            measured = true_value
        
        return measured, self.properties.readout_fidelity
    
    def get_coherence(self, current_time_ns: float) -> Tuple[float, float]:
        """Get current coherence based on time since last reset"""
        time_since_reset = current_time_ns - self.last_operation_time
        t1_coherence = np.exp(-time_since_reset / self.properties.t1_time)
        t2_coherence = np.exp(-time_since_reset / self.properties.t2_time)
        return t1_coherence, t2_coherence

class QuantumProcessor:
    """Manages a collection of virtual qubits"""
    def __init__(self, num_qubits: int = 32):
        self.num_qubits = num_qubits
        self.qubits = [VirtualQubit(i) for i in range(num_qubits)]
        self.coupling_map = self._generate_coupling_map()
        self.current_time_ns = 0
        self.gate_times = {
            'H': 50, 'X': 50, 'Y': 50, 'Z': 50,
            'S': 35, 'T': 35,
            'CNOT': 100, 'CZ': 120, 'SWAP': 150
        }
    
    def _generate_coupling_map(self) -> List[Tuple[int, int]]:
        """Generate nearest-neighbor coupling map"""
        cmap = []
        for i in range(self.num_qubits - 1):
            cmap.append((i, i + 1))
        return cmap
    
    def execute_gate(self, gate_type: str, targets: List[int], 
                    controls: Optional[List[int]] = None) -> float:
        """Execute quantum gate and return total error probability"""
        total_error = 0.0
        gate_time = self.gate_times.get(gate_type, 50)
        
        # Update all involved qubits
        involved_qubits = targets
        if controls:
            involved_qubits.extend(controls)
        
        for qubit_idx in involved_qubits:
            if qubit_idx < len(self.qubits):
                if gate_type in ['CNOT', 'CZ', 'SWAP']:
                    gate_class = 'two_qubit'
                else:
                    gate_class = 'single'
                
                error = self.qubits[qubit_idx].apply_gate(gate_class, gate_time)
                total_error = max(total_error, error)
        
        self.current_time_ns += gate_time
        return total_error
    
    def get_processor_fidelity(self) -> float:
        """Calculate overall processor fidelity"""
        fidelities = []
        for qubit in self.qubits:
            # Include all error sources
            t1, t2 = qubit.get_coherence(self.current_time_ns)
            readout_fid = qubit.properties.readout_fidelity
            avg_gate_fid = np.mean(list(qubit.properties.gate_fidelity.values()))
            
            qubit_fidelity = t1 * t2 * readout_fid * avg_gate_fid
            fidelities.append(qubit_fidelity)
        
        return np.prod(fidelities) ** (1/len(fidelities))
    
    def measure_all(self) -> Tuple[List[int], float]:
        """Measure all qubits and return overall fidelity"""
        measurements = []
        fidelities = []
        
        for qubit in self.qubits:
            result, fidelity = qubit.measure()
            measurements.append(result)
            fidelities.append(fidelity)
        
        return measurements, np.mean(fidelities)
    
    def reset_qubit(self, qubit_idx: int):
        """Reset qubit to ground state"""
        if qubit_idx < len(self.qubits):
            self.qubits[qubit_idx].state = QubitState.GROUND
            self.qubits[qubit_idx].last_operation_time = self.current_time_ns
    
    def get_connectivity_graph(self) -> Dict:
        """Get processor connectivity information"""
        return {
            'num_qubits': self.num_qubits,
            'coupling_map': self.coupling_map,
            'degree_distribution': self._calculate_degree_distribution(),
            'diameter': self._calculate_graph_diameter()
        }
    
    def _calculate_degree_distribution(self) -> List[int]:
        """Calculate degree of each qubit in coupling map"""
        degrees = [0] * self.num_qubits
        for i, j in self.coupling_map:
            if i < self.num_qubits:
                degrees[i] += 1
            if j < self.num_qubits:
                degrees[j] += 1
        return degrees
    
    def _calculate_graph_diameter(self) -> int:
        """Calculate diameter of coupling graph"""
        # Simplified calculation for linear chain
        return self.num_qubits - 1