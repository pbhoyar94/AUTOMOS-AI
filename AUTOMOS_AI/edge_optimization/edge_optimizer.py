"""
AUTOMOS AI Edge Optimizer
Edge deployment optimization and performance tuning
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import psutil
import platform
import subprocess
import json

logger = logging.getLogger(__name__)

class EdgeOptimizer:
    """Edge deployment optimizer"""
    
    def __init__(self):
        """Initialize edge optimizer"""
        logger.info("Initializing Edge Optimizer...")
        
        # System information
        self.system_info = self._get_system_info()
        
        # Optimization profiles
        self.optimization_profiles = {
            'jetson_nano': {
                'max_memory_mb': 4096,
                'cpu_cores': 4,
                'gpu_memory_mb': 2048,
                'recommended_batch_size': 1,
                'precision': 'int8',
                'thread_count': 4
            },
            'jetson_xavier': {
                'max_memory_mb': 32768,
                'cpu_cores': 8,
                'gpu_memory_mb': 16384,
                'recommended_batch_size': 2,
                'precision': 'mixed',
                'thread_count': 8
            },
            'raspberry_pi_4': {
                'max_memory_mb': 4096,
                'cpu_cores': 4,
                'gpu_memory_mb': 0,
                'recommended_batch_size': 1,
                'precision': 'int8',
                'thread_count': 4
            },
            'industrial_pc': {
                'max_memory_mb': 16384,
                'cpu_cores': 8,
                'gpu_memory_mb': 4096,
                'recommended_batch_size': 4,
                'precision': 'fp16',
                'thread_count': 8
            }
        }
        
        # Optimization results
        self.optimization_results = {}
        
        logger.info("Edge Optimizer initialized")
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        
        system_info = {
            'platform': platform.system(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0],
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        # Try to detect GPU
        try:
            import torch
            if torch.cuda.is_available():
                system_info['gpu_available'] = True
                system_info['gpu_name'] = torch.cuda.get_device_name(0)
                system_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
            else:
                system_info['gpu_available'] = False
        except ImportError:
            system_info['gpu_available'] = False
        
        # Detect edge device type
        system_info['device_type'] = self._detect_device_type(system_info)
        
        logger.info(f"Detected system: {system_info['device_type']}")
        return system_info
    
    def _detect_device_type(self, system_info: Dict) -> str:
        """Detect edge device type"""
        
        # Simple heuristics for device detection
        if 'jetson' in system_info.get('processor', '').lower():
            if 'nano' in system_info.get('processor', '').lower():
                return 'jetson_nano'
            else:
                return 'jetson_xavier'
        elif 'arm' in system_info.get('architecture', '').lower():
            if system_info.get('memory_total', 0) < 8 * 1024**3:  # Less than 8GB
                return 'raspberry_pi_4'
        
        return 'industrial_pc'
    
    def optimize_for_edge(self, deployment_config: Dict) -> Dict:
        """
        Optimize system for edge deployment
        
        Args:
            deployment_config: Deployment configuration
            
        Returns:
            Dict: Optimization results
        """
        
        logger.info("Optimizing for edge deployment")
        
        try:
            device_type = self.system_info['device_type']
            profile = self.optimization_profiles.get(device_type, self.optimization_profiles['industrial_pc'])
            
            # Apply memory optimizations
            memory_optimizations = self._optimize_memory(profile)
            
            # Apply CPU optimizations
            cpu_optimizations = self._optimize_cpu(profile)
            
            # Apply GPU optimizations if available
            gpu_optimizations = {}
            if self.system_info.get('gpu_available', False):
                gpu_optimizations = self._optimize_gpu(profile)
            
            # Apply system optimizations
            system_optimizations = self._optimize_system(profile)
            
            # Generate optimization report
            optimization_report = {
                'device_type': device_type,
                'system_info': self.system_info,
                'profile_used': profile,
                'memory_optimizations': memory_optimizations,
                'cpu_optimizations': cpu_optimizations,
                'gpu_optimizations': gpu_optimizations,
                'system_optimizations': system_optimizations,
                'estimated_performance_gain': self._estimate_performance_gain(profile),
                'optimization_timestamp': psutil.boot_time()
            }
            
            self.optimization_results[device_type] = optimization_report
            
            logger.info(f"Edge optimization completed for {device_type}")
            return optimization_report
            
        except Exception as e:
            logger.error(f"Edge optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_memory(self, profile: Dict) -> Dict:
        """Optimize memory usage"""
        
        logger.info("Optimizing memory usage")
        
        optimizations = {}
        
        # Set memory limits
        max_memory_mb = profile['max_memory_mb']
        
        # Calculate memory allocations
        memory_allocations = {
            'reasoning_engine_mb': int(max_memory_mb * 0.3),
            'perception_pipeline_mb': int(max_memory_mb * 0.4),
            'safety_critic_mb': int(max_memory_mb * 0.1),
            'world_model_mb': int(max_memory_mb * 0.15),
            'system_overhead_mb': int(max_memory_mb * 0.05)
        }
        
        optimizations['memory_allocations'] = memory_allocations
        
        # Memory optimization settings
        optimizations['settings'] = {
            'enable_memory_mapping': True,
            'use_shared_memory': True,
            'garbage_collection_frequency': 'high',
            'memory_pool_size_mb': int(max_memory_mb * 0.1),
            'max_cache_size_mb': int(max_memory_mb * 0.05)
        }
        
        return optimizations
    
    def _optimize_cpu(self, profile: Dict) -> Dict:
        """Optimize CPU usage"""
        
        logger.info("Optimizing CPU usage")
        
        optimizations = {}
        
        # Thread optimization
        cpu_cores = profile['cpu_cores']
        thread_allocations = {
            'perception_threads': max(2, cpu_cores // 2),
            'reasoning_threads': max(1, cpu_cores // 4),
            'safety_threads': 1,  # Safety critic needs dedicated thread
            'world_model_threads': max(1, cpu_cores // 4),
            'system_threads': 1
        }
        
        optimizations['thread_allocations'] = thread_allocations
        
        # CPU optimization settings
        optimizations['settings'] = {
            'cpu_affinity': True,
            'process_priority': 'high',
            'thread_priority': {
                'safety_critic': 'realtime',
                'perception_pipeline': 'high',
                'reasoning_engine': 'normal',
                'world_model': 'normal'
            },
            'cpu_governor': 'performance',
            'enable_cpu_scaling': False
        }
        
        return optimizations
    
    def _optimize_gpu(self, profile: Dict) -> Dict:
        """Optimize GPU usage"""
        
        logger.info("Optimizing GPU usage")
        
        optimizations = {}
        
        gpu_memory_mb = profile.get('gpu_memory_mb', 0)
        
        # GPU memory allocations
        memory_allocations = {
            'perception_gpu_mb': int(gpu_memory_mb * 0.6),
            'reasoning_gpu_mb': int(gpu_memory_mb * 0.3),
            'safety_gpu_mb': int(gpu_memory_mb * 0.05),
            'system_gpu_mb': int(gpu_memory_mb * 0.05)
        }
        
        optimizations['memory_allocations'] = memory_allocations
        
        # GPU optimization settings
        optimizations['settings'] = {
            'precision': profile['precision'],
            'batch_size': profile['recommended_batch_size'],
            'enable_tensorrt': True,
            'enable_cuda_graphs': True,
            'memory_pool_size_mb': int(gpu_memory_mb * 0.1),
            'max_workspace_size_mb': int(gpu_memory_mb * 0.2)
        }
        
        return optimizations
    
    def _optimize_system(self, profile: Dict) -> Dict:
        """Optimize system settings"""
        
        logger.info("Optimizing system settings")
        
        optimizations = {}
        
        # System optimization settings
        optimizations['settings'] = {
            'disable_swap': True,
            'optimize_disk_io': True,
            'enable_realtime_scheduling': True,
            'set_cpu_governor': 'performance',
            'disable_unnecessary_services': True,
            'optimize_network_settings': True
        }
        
        # Power management
        optimizations['power_management'] = {
            'power_mode': 'performance',
            'disable_sleep': True,
            'thermal_management': 'aggressive'
        }
        
        return optimizations
    
    def _estimate_performance_gain(self, profile: Dict) -> Dict:
        """Estimate performance gains from optimization"""
        
        # Mock performance estimates
        device_type = self.system_info['device_type']
        
        performance_gains = {
            'jetson_nano': {
                'inference_speedup': 2.5,
                'memory_efficiency': 0.7,
                'power_savings': 0.3,
                'thermal_improvement': 0.4
            },
            'jetson_xavier': {
                'inference_speedup': 3.0,
                'memory_efficiency': 0.8,
                'power_savings': 0.4,
                'thermal_improvement': 0.5
            },
            'raspberry_pi_4': {
                'inference_speedup': 1.8,
                'memory_efficiency': 0.6,
                'power_savings': 0.2,
                'thermal_improvement': 0.3
            },
            'industrial_pc': {
                'inference_speedup': 2.0,
                'memory_efficiency': 0.7,
                'power_savings': 0.1,
                'thermal_improvement': 0.2
            }
        }
        
        return performance_gains.get(device_type, performance_gains['industrial_pc'])
    
    def apply_optimizations(self, optimization_report: Dict) -> bool:
        """Apply system optimizations"""
        
        logger.info("Applying system optimizations")
        
        try:
            # Apply system optimizations
            system_settings = optimization_report.get('system_optimizations', {}).get('settings', {})
            
            # Apply CPU optimizations
            self._apply_cpu_optimizations(optimization_report.get('cpu_optimizations', {}))
            
            # Apply memory optimizations
            self._apply_memory_optimizations(optimization_report.get('memory_optimizations', {}))
            
            # Apply GPU optimizations if available
            if self.system_info.get('gpu_available', False):
                self._apply_gpu_optimizations(optimization_report.get('gpu_optimizations', {}))
            
            logger.info("System optimizations applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            return False
    
    def _apply_cpu_optimizations(self, cpu_optimizations: Dict):
        """Apply CPU optimizations"""
        
        try:
            # Set CPU governor to performance
            if self.system_info['platform'] == 'Linux':
                subprocess.run(['sudo', 'cpupower', 'frequency-set', '-g', 'performance'], 
                             check=False, capture_output=True)
            
            logger.info("CPU optimizations applied")
            
        except Exception as e:
            logger.warning(f"Failed to apply CPU optimizations: {e}")
    
    def _apply_memory_optimizations(self, memory_optimizations: Dict):
        """Apply memory optimizations"""
        
        try:
            # Disable swap if requested
            if self.system_info['platform'] == 'Linux':
                subprocess.run(['sudo', 'swapoff', '-a'], check=False, capture_output=True)
            
            logger.info("Memory optimizations applied")
            
        except Exception as e:
            logger.warning(f"Failed to apply memory optimizations: {e}")
    
    def _apply_gpu_optimizations(self, gpu_optimizations: Dict):
        """Apply GPU optimizations"""
        
        try:
            import torch
            
            if torch.cuda.is_available():
                # Set GPU memory fraction
                settings = gpu_optimizations.get('settings', {})
                memory_allocations = gpu_optimizations.get('memory_allocations', {})
                
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
                perception_memory = memory_allocations.get('perception_gpu_mb', 0) * 1024 * 1024
                
                if perception_memory > 0:
                    torch.cuda.set_per_process_memory_fraction(
                        perception_memory / total_gpu_memory
                    )
                
                logger.info("GPU optimizations applied")
            
        except Exception as e:
            logger.warning(f"Failed to apply GPU optimizations: {e}")
    
    def generate_deployment_script(self, optimization_report: Dict, output_path: str) -> str:
        """Generate deployment script"""
        
        logger.info("Generating deployment script")
        
        script_content = f"""#!/bin/bash
# AUTOMOS AI Edge Deployment Script
# Generated for {self.system_info['device_type']}

set -e

echo "Starting AUTOMOS AI Edge Deployment..."

# Apply system optimizations
echo "Applying system optimizations..."

# CPU optimizations
echo "Setting CPU governor to performance..."
sudo cpupower frequency-set -g performance || echo "CPU governor setting failed"

# Memory optimizations
echo "Configuring memory settings..."
sudo swapoff -a || echo "Swap disable failed"
echo 1 | sudo tee /proc/sys/vm/drop_caches || echo "Cache clear failed"

# GPU optimizations (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "Configuring GPU settings..."
    nvidia-smi -pm 1 || echo "GPU persistence mode failed"
fi

# Set environment variables
export AUTOMOS_DEVICE_TYPE="{self.system_info['device_type']}"
export AUTOMOS_CPU_CORES={self.system_info['cpu_count']}
export AUTOMOS_MEMORY_GB=$(( {self.system_info['memory_total']} / 1024 / 1024 / 1024 ))

# Start AUTOMOS AI
echo "Starting AUTOMOS AI..."
python3 main.py --mode deployment --hardware edge

echo "AUTOMOS AI deployment completed!"
"""
        
        # Save script
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(output_path, 0o755)
        
        logger.info(f"Deployment script saved to {output_path}")
        return output_path
    
    def get_optimization_results(self) -> Dict:
        """Get optimization results"""
        return self.optimization_results.copy()
    
    def benchmark_system(self) -> Dict:
        """Benchmark system performance"""
        
        logger.info("Running system benchmark")
        
        benchmark_results = {
            'device_type': self.system_info['device_type'],
            'cpu_benchmark': self._benchmark_cpu(),
            'memory_benchmark': self._benchmark_memory(),
            'disk_benchmark': self._benchmark_disk()
        }
        
        if self.system_info.get('gpu_available', False):
            benchmark_results['gpu_benchmark'] = self._benchmark_gpu()
        
        return benchmark_results
    
    def _benchmark_cpu(self) -> Dict:
        """Benchmark CPU performance"""
        
        import time
        
        # Simple CPU benchmark
        start_time = time.time()
        
        # Perform CPU-intensive task
        result = sum(i * i for i in range(1000000))
        
        cpu_time = time.time() - start_time
        
        return {
            'computation_time': cpu_time,
            'operations_per_second': 1000000 / cpu_time,
            'cpu_utilization': psutil.cpu_percent(interval=1)
        }
    
    def _benchmark_memory(self) -> Dict:
        """Benchmark memory performance"""
        
        import time
        
        # Simple memory benchmark
        start_time = time.time()
        
        # Allocate and access memory
        data = [i for i in range(1000000)]
        _ = sum(data)
        
        memory_time = time.time() - start_time
        
        memory_info = psutil.virtual_memory()
        
        return {
            'allocation_time': memory_time,
            'memory_speed_mb_per_second': (1000000 * 8) / (1024 * 1024 * memory_time),  # Rough estimate
            'available_memory_gb': memory_info.available / (1024**3),
            'memory_utilization': memory_info.percent
        }
    
    def _benchmark_disk(self) -> Dict:
        """Benchmark disk performance"""
        
        import tempfile
        import time
        
        # Simple disk benchmark
        with tempfile.NamedTemporaryFile(delete=True) as f:
            # Write test
            data = b'x' * (1024 * 1024)  # 1MB
            start_time = time.time()
            f.write(data * 100)  # 100MB
            f.flush()
            write_time = time.time() - start_time
            
            # Read test
            f.seek(0)
            start_time = time.time()
            _ = f.read()
            read_time = time.time() - start_time
        
        return {
            'write_speed_mb_per_second': 100 / write_time,
            'read_speed_mb_per_second': 100 / read_time,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    
    def _benchmark_gpu(self) -> Dict:
        """Benchmark GPU performance"""
        
        try:
            import torch
            import time
            
            if not torch.cuda.is_available():
                return {'error': 'CUDA not available'}
            
            # Simple GPU benchmark
            device = torch.device('cuda')
            
            # Create tensors
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            # Warmup
            for _ in range(10):
                _ = torch.mm(x, y)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = torch.mm(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            return {
                'matrix_multiplication_time': gpu_time / 100,
                'operations_per_second': (2 * 1000**3) / (gpu_time / 100),  # FLOPs
                'gpu_memory_utilization': torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            }
            
        except Exception as e:
            return {'error': str(e)}
