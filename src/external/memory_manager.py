#!/usr/bin/env python3
"""
Memory Manager for Quantum Simulations
"""

import numpy as np
import psutil
from typing import Dict, List, Optional, Any

class MemoryManager:
    """Manage memory allocation for quantum simulations"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.allocated_blocks = {}
        self.block_counter = 0
    
    def allocate(self, size_gb: float, purpose: str = "quantum_state") -> int:
        """Allocate memory block"""
        available = psutil.virtual_memory().available / 1e9
        
        if size_gb > available:
            raise MemoryError(f"Requested {size_gb:.2f} GB, only {available:.2f} GB available")
        
        if size_gb > self.max_memory_gb:
            raise MemoryError(f"Requested {size_gb:.2f} GB exceeds limit {self.max_memory_gb:.2f} GB")
        
        block_id = self.block_counter
        self.allocated_blocks[block_id] = {
            'size_gb': size_gb,
            'purpose': purpose,
            'status': 'allocated'
        }
        self.block_counter += 1
        
        return block_id
    
    def free(self, block_id: int):
        """Free memory block"""
        if block_id in self.allocated_blocks:
            del self.allocated_blocks[block_id]
    
    def get_usage(self) -> Dict:
        """Get memory usage statistics"""
        total_allocated = sum(block['size_gb'] for block in self.allocated_blocks.values())
        available = psutil.virtual_memory().available / 1e9
        
        return {
            'total_allocated_gb': total_allocated,
            'available_gb': available,
            'max_limit_gb': self.max_memory_gb,
            'blocks': len(self.allocated_blocks),
            'usage_percentage': (total_allocated / self.max_memory_gb) * 100
        }
    
    def clear_all(self):
        """Clear all allocated memory"""
        self.allocated_blocks.clear()
