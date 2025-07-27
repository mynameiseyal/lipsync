#!/usr/bin/env python3
"""
GPU Utilities for Multi-Speaker Video Generation
Handles GPU detection, device management, and CPU fallback.
"""

import os
import logging
import torch
import warnings
from typing import Dict, Optional, Tuple

class GPUManager:
    """Manages GPU device selection and fallback to CPU."""
    
    def __init__(self, config: Dict):
        """Initialize GPU manager with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.device_name = None
        self.memory_available = 0
        self.is_gpu_available = False
        
        # Initialize device
        self._detect_and_setup_device()
    
    def _detect_and_setup_device(self):
        """Detect available GPU and setup device."""
        try:
            # Check if MPS (Apple Silicon) is available first
            if torch.backends.mps.is_available() and self.config['gpu_settings']['enabled']:
                self.is_gpu_available = True
                self.device = torch.device('mps')
                self.device_name = "Apple M3 Pro (MPS)"
                self.memory_available = 0  # MPS doesn't expose memory info easily
                
                self.logger.info(f"[MPS] Apple Silicon GPU initialized: {self.device_name}")
                self.logger.info(f"   Device: {self.device}")
                self.logger.info(f"   Using Metal Performance Shaders")
                
            # Check if CUDA is available
            elif torch.cuda.is_available() and self.config['gpu_settings']['enabled']:
                self.is_gpu_available = True
                
                # Get device count and select device
                device_count = torch.cuda.device_count()
                self.logger.info(f"Found {device_count} CUDA device(s)")
                
                # Select best GPU or use specified device
                device_id = self._select_best_gpu()
                self.device = torch.device(f'cuda:{device_id}')
                
                # Get device properties
                props = torch.cuda.get_device_properties(device_id)
                self.device_name = props.name
                self.memory_available = props.total_memory
                
                # Set memory management
                if self.config['gpu_settings']['allow_growth']:
                    torch.cuda.empty_cache()
                
                # Set memory fraction if specified
                memory_fraction = self.config['gpu_settings'].get('memory_fraction', 0.8)
                if memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(memory_fraction, device_id)
                
                self.logger.info(f"[CUDA] GPU initialized: {self.device_name}")
                self.logger.info(f"   Device: {self.device}")
                self.logger.info(f"   Memory: {self.memory_available / 1e9:.1f} GB")
                
            else:
                self._fallback_to_cpu()
                
        except Exception as e:
            self.logger.warning(f"GPU initialization failed: {e}")
            self._fallback_to_cpu()
    
    def _select_best_gpu(self) -> int:
        """Select the best available GPU device."""
        device_setting = self.config['gpu_settings'].get('device', 'auto')
        
        if device_setting == 'auto':
            # Select GPU with most available memory
            best_device = 0
            max_memory = 0
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory = props.total_memory
                
                # Check memory usage
                torch.cuda.set_device(i)
                memory_used = torch.cuda.memory_allocated()
                memory_available = memory - memory_used
                
                if memory_available > max_memory:
                    max_memory = memory_available
                    best_device = i
            
            return best_device
        else:
            # Use specified device
            device_id = int(device_setting)
            if device_id >= torch.cuda.device_count():
                self.logger.warning(f"Specified GPU {device_id} not available, using GPU 0")
                return 0
            return device_id
    
    def _fallback_to_cpu(self):
        """Fallback to CPU processing."""
        self.is_gpu_available = False
        self.device = torch.device('cpu')
        self.device_name = "CPU"
        
        if self.config['gpu_settings'].get('fallback_to_cpu', True):
            self.logger.info("[CPU] Using CPU for processing (GPU not available or disabled)")
        else:
            raise RuntimeError("GPU required but not available, and CPU fallback is disabled")
    
    def get_device(self) -> torch.device:
        """Get the current processing device."""
        return self.device
    
    def get_device_info(self) -> Dict:
        """Get device information."""
        return {
            'device': str(self.device),
            'device_name': self.device_name,
            'is_gpu': self.is_gpu_available,
            'memory_available': self.memory_available,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available()
        }
    
    def get_optimal_batch_size(self, base_batch_size: int = None) -> int:
        """Get optimal batch size based on available memory."""
        if base_batch_size is None:
            base_batch_size = self.config['gpu_settings'].get('batch_size', 4)
        
        if not self.is_gpu_available:
            # For CPU, use smaller batch sizes
            return max(1, base_batch_size // 2)
        
        # For GPU, adjust based on available memory
        available_memory = self.memory_available
        if available_memory > 8e9:  # 8GB+
            return base_batch_size
        elif available_memory > 4e9:  # 4GB+
            return max(1, base_batch_size // 2)
        else:  # Less than 4GB
            return 1
    
    def setup_environment_variables(self):
        """Setup environment variables for optimal GPU performance."""
        if self.is_gpu_available:
            # CUDA optimizations
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['TORCH_CUDA_ARCH_LIST'] = '6.0;6.1;7.0;7.5;8.0;8.6'
            
            # Memory management
            if self.config['gpu_settings'].get('memory_efficient', True):
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Mixed precision
            if self.config['gpu_settings'].get('mixed_precision', True):
                os.environ['TORCH_AUTOCAST_ENABLED'] = '1'
        
        # CPU optimizations
        if self.config['gpu_settings'].get('num_workers', 4) > 1:
            os.environ['OMP_NUM_THREADS'] = str(self.config['gpu_settings']['num_workers'])
    
    def cleanup(self):
        """Cleanup GPU resources."""
        if self.is_gpu_available:
            if str(self.device) == 'mps':
                # MPS cleanup (if needed)
                self.logger.info("🧹 MPS resources cleaned")
            else:
                # CUDA cleanup
                torch.cuda.empty_cache()
                self.logger.info("🧹 GPU memory cache cleared")
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current GPU memory usage in GB."""
        if self.is_gpu_available:
            if str(self.device) == 'mps':
                # MPS doesn't expose memory info easily, return estimates
                return 0.0, 0.0
            else:
                # CUDA memory info
                allocated = torch.cuda.memory_allocated() / 1e9
                cached = torch.cuda.memory_reserved() / 1e9
                return allocated, cached
        return 0.0, 0.0
    
    def monitor_memory(self, operation_name: str = ""):
        """Log current memory usage."""
        if self.is_gpu_available:
            allocated, cached = self.get_memory_usage()
            total = self.memory_available / 1e9
            self.logger.debug(f"GPU Memory {operation_name}: {allocated:.1f}GB allocated, "
                            f"{cached:.1f}GB cached, {total:.1f}GB total")


def get_gpu_manager(config: Dict) -> GPUManager:
    """Factory function to create GPU manager."""
    return GPUManager(config)


def setup_gpu_environment(config: Dict) -> GPUManager:
    """Setup GPU environment and return manager."""
    # Suppress some CUDA warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    
    # Create GPU manager
    gpu_manager = GPUManager(config)
    
    # Setup environment
    gpu_manager.setup_environment_variables()
    
    return gpu_manager 