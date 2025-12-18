"""
Configuration for QNVM - Minimal version
"""

from dataclasses import dataclass, asdict
from typing import Dict
from enum import Enum

class BackendType(Enum):
    """Supported backends"""
    TENSOR_NETWORK = "tensor_network"
    QISKIT = "qiskit"
    CIRQ = "cirq"
    INTERNAL = "internal"

class CompressionMethod(Enum):
    """Compression methods"""
    AUTO = "auto"
    TOP_K = "top_k"
    THRESHOLD = "threshold"

@dataclass
class QNVMConfig:
    """Minimal configuration"""
    max_qubits: int = 4
    max_memory_gb: float = 2.0
    backend: BackendType = BackendType.INTERNAL
    error_correction: bool = False
    compression_enabled: bool = False
    compression_method: CompressionMethod = CompressionMethod.AUTO
    compression_ratio: float = 0.1
    validation_enabled: bool = True
    log_level: str = "INFO"
    
    def to_dict(self):
        return asdict(self)
