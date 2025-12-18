from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
from enum import Enum

class BackendType(Enum):
    """Supported quantum backends"""
    TENSOR_NETWORK = "tensor_network"
    QISKIT = "qiskit"
    CIRQ = "cirq"
    STABILIZER = "stabilizer"
    INTERNAL = "internal"

class CompressionMethod(Enum):
    """State compression methods"""
    AUTO = "auto"
    TOP_K = "top_k"
    THRESHOLD = "threshold"
    WAVELET = "wavelet"
    SVD = "svd"

@dataclass
class QNVMConfig:
    """Main configuration for QNVM"""
    # Core settings
    max_qubits: int = 32
    max_memory_gb: float = 8.0
    
    # Backend configuration
    backend: BackendType = BackendType.TENSOR_NETWORK
    backend_options: Dict = None
    
    # Error correction
    error_correction: bool = False
    code_type: str = "surface_code"
    code_distance: int = 3
    
    # Compression settings
    compression_enabled: bool = True
    compression_method: CompressionMethod = CompressionMethod.AUTO
    compression_ratio: float = 0.1
    sparse_threshold: float = 1e-6
    
    # Validation settings
    validation_enabled: bool = True
    ground_truth_verification: bool = False
    validation_tolerance: float = 1e-10
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 512
    parallel_execution: bool = False
    num_threads: int = 4
    
    # Logging settings
    log_level: str = "INFO"
    enable_telemetry: bool = True
    telemetry_interval: int = 1000
    
    def __post_init__(self):
        """Initialize default values"""
        if self.backend_options is None:
            self.backend_options = {
                'qiskit': {'optimization_level': 3, 'shots': 1024},
                'cirq': {'noise_model': 'depolarizing', 'repetitions': 1000},
                'tensor_network': {'bond_dimension': 4, 'contraction_optimizer': 'greedy'}
            }
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'QNVMConfig':
        """Create config from dictionary"""
        # Convert string enums back to Enum types
        if 'backend' in config_dict and isinstance(config_dict['backend'], str):
            config_dict['backend'] = BackendType(config_dict['backend'])
        
        if 'compression_method' in config_dict and isinstance(config_dict['compression_method'], str):
            config_dict['compression_method'] = CompressionMethod(config_dict['compression_method'])
        
        return cls(**config_dict)
    
    def validate(self) -> List[str]:
        """Validate configuration parameters"""
        errors = []
        
        if self.max_qubits > 32:
            errors.append(f"max_qubits ({self.max_qubits}) exceeds maximum supported value (32)")
        
        if self.max_memory_gb < 0.1:
            errors.append(f"max_memory_gb ({self.max_memory_gb}) must be at least 0.1 GB")
        
        if not 0 < self.compression_ratio <= 1.0:
            errors.append(f"compression_ratio ({self.compression_ratio}) must be between 0 and 1")
        
        if self.code_distance not in [3, 5, 7]:
            errors.append(f"code_distance ({self.code_distance}) must be 3, 5, or 7")
        
        return errors
    
    def get_memory_requirements(self) -> Dict:
        """Calculate memory requirements based on configuration"""
        dense_size_gb = (2 ** self.max_qubits) * 16 / (1024**3)  # complex128
        
        if self.compression_enabled:
            estimated_size = dense_size_gb * self.compression_ratio
        else:
            estimated_size = dense_size_gb
        
        return {
            'dense_size_gb': dense_size_gb,
            'estimated_size_gb': estimated_size,
            'compression_factor': dense_size_gb / estimated_size if estimated_size > 0 else 1.0,
            'within_budget': estimated_size <= self.max_memory_gb
        }