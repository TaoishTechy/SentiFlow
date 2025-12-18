# src/quantum_core_nexus/utils/numpy_encoder.py
"""
Custom JSON encoder for numpy types
"""

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.complexfloating):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)
