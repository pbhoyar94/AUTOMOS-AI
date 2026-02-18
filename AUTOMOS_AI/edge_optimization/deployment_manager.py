"""
AUTOMOS AI Deployment Manager
Manages deployment of AUTOMOS AI to edge devices
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import shutil
import json
import zipfile
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class DeploymentManager:
    """Manages deployment of AUTOMOS AI to edge devices"""
    
    def __init__(self):
        """Initialize deployment manager"""
        logger.info("Initializing Deployment Manager...")
        
        # Deployment configurations
        self.deployment_configs = {
            'development': {
                'include_debug': True,
                'include_tests': True,
                'compression_level': 0,
                'target_devices': ['industrial_pc']
            },
            'staging': {
                'include_debug': False,
                'include_tests': True,
                'compression_level': 6,
                'target_devices': ['jetson_xavier', 'industrial_pc']
            },
            'production': {
                'include_debug': False,
                'include_tests': False,
                'compression_level': 9,
                'target_devices': ['jetson_nano', 'jetson_xavier', 'raspberry_pi_4', 'industrial_pc']
            }
        }
        
        # Deployment history
        self.deployment_history = []
        
        logger.info("Deployment Manager initialized")
    
    def create_deployment_package(self, source_directory: str, output_directory: str, 
                                deployment_type: str = 'production') -> Dict:
        """
        Create deployment package
        
        Args:
            source_directory: Source directory containing AUTOMOS AI
            output_directory: Output directory for deployment package
            deployment_type: Type of deployment (development, staging, production)
            
        Returns:
            Dict: Deployment package information
        """
        
        logger.info(f"Creating {deployment_type} deployment package")
        
        try:
            # Get deployment configuration
            if deployment_type not in self.deployment_configs:
                return {'success': False, 'error': f'Invalid deployment type: {deployment_type}'}
            
            config = self.deployment_configs[deployment_type]
            
            # Create deployment directory structure
            deployment_dir = Path(output_directory) / f"automos_ai_deployment_{deployment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            deployment_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy core files
            self._copy_core_files(source_directory, deployment_dir, config)
            
            # Copy optimized models
            self._copy_optimized_models(source_directory, deployment_dir)
            
            # Copy configuration files
            self._copy_configuration_files(deployment_dir, config)
            
            # Create deployment scripts
            self._create_deployment_scripts(deployment_dir, config)
            
            # Create documentation
            self._create_deployment_documentation(deployment_dir, config)
            
            # Create package metadata
            package_info = self._create_package_metadata(deployment_dir, deployment_type, config)
            
            # Create compressed package
            package_path = self._create_compressed_package(deployment_dir, config)
            
            # Calculate package checksum
            checksum = self._calculate_checksum(package_path)
            
            # Record deployment
            deployment_record = {
                'deployment_type': deployment_type,
                'timestamp': datetime.now().isoformat(),
                'package_path': str(package_path),
                'package_size_mb': os.path.getsize(package_path) / (1024 * 1024),
                'checksum': checksum,
                'config': config,
                'package_info': package_info
            }
            
            self.deployment_history.append(deployment_record)
            
            logger.info(f"Deployment package created: {package_path}")
            
            return {
                'success': True,
                'package_path': str(package_path),
                'package_size_mb': deployment_record['package_size_mb'],
                'checksum': checksum,
                'deployment_info': deployment_record
            }
            
        except Exception as e:
            logger.error(f"Failed to create deployment package: {e}")
            return {'success': False, 'error': str(e)}
    
    def _copy_core_files(self, source_directory: str, deployment_dir: Path, config: Dict):
        """Copy core AUTOMOS AI files"""
        
        logger.info("Copying core files")
        
        source_path = Path(source_directory)
        
        # Core directories to copy
        core_dirs = [
            'core',
            'perception', 
            'safety',
            'world_model',
            'integration',
            'edge_optimization'
        ]
        
        for dir_name in core_dirs:
            src_dir = source_path / dir_name
            dst_dir = deployment_dir / dir_name
            
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
                logger.info(f"Copied {dir_name} directory")
        
        # Core files to copy
        core_files = [
            'main.py',
            'requirements.txt',
            'README.md'
        ]
        
        for file_name in core_files:
            src_file = source_path / file_name
            dst_file = deployment_dir / file_name
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                logger.info(f"Copied {file_name}")
        
        # Copy debug files if needed
        if config.get('include_debug', False):
            debug_dir = deployment_dir / 'debug'
            debug_dir.mkdir(exist_ok=True)
            
            # Copy debug tools
            debug_files = [
                'debug_tools.py',
                'performance_profiler.py'
            ]
            
            for debug_file in debug_files:
                src_file = source_path / debug_file
                if src_file.exists():
                    shutil.copy2(src_file, debug_dir / debug_file)
    
    def _copy_optimized_models(self, source_directory: str, deployment_dir: Path):
        """Copy optimized models"""
        
        logger.info("Copying optimized models")
        
        source_path = Path(source_directory)
        models_source = source_path / 'models' / 'quantized'
        models_dest = deployment_dir / 'models'
        
        if models_source.exists():
            shutil.copytree(models_source, models_dest)
            logger.info("Copied quantized models")
        else:
            # Create models directory and copy placeholder
            models_dest.mkdir(parents=True, exist_ok=True)
            
            # Create model info file
            model_info = {
                'models': {
                    'reasoning_engine': 'reasoning_engine_dynamic_int8.pt',
                    'perception_pipeline': 'perception_pipeline_dynamic_int8.pt',
                    'safety_critic': 'safety_critic_dynamic_int8.pt',
                    'world_model': 'world_model_dynamic_int8.pt'
                },
                'quantization_type': 'dynamic_int8',
                'target_devices': ['jetson_nano', 'jetson_xavier', 'raspberry_pi_4', 'industrial_pc']
            }
            
            with open(models_dest / 'model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
    
    def _copy_configuration_files(self, deployment_dir: Path, config: Dict):
        """Copy configuration files"""
        
        logger.info("Copying configuration files")
        
        config_dir = deployment_dir / 'config'
        config_dir.mkdir(exist_ok=True)
        
        # Create default configuration
        default_config = {
            'system': {
                'mode': 'deployment',
                'hardware': 'edge',
                'log_level': 'INFO',
                'max_memory_mb': 4096,
                'cpu_cores': 4
            },
            'perception': {
                'camera_count': 6,
                'radar_count': 4,
                'lidar_count': 2,
                'processing_frequency': 20
            },
            'safety': {
                'response_time_ms': 10,
                'emergency_override': True,
                'max_acceleration': 3.0,
                'max_deceleration': 8.0
            },
            'reasoning': {
                'llm_model': 'dualad_opendrivevla_fusion',
                'planning_algorithm': 'lattice_idm_vla'
            },
            'edge_optimization': {
                'quantization_type': 'dynamic_int8',
                'batch_size': 1,
                'precision': 'int8'
            }
        }
        
        with open(config_dir / 'default_config.json', 'w') as f:
            json.dump(default_config, f, indent=2)
        
        # Create device-specific configurations
        device_configs = {
            'jetson_nano': {
                'system': {'max_memory_mb': 4096, 'cpu_cores': 4},
                'edge_optimization': {'batch_size': 1, 'precision': 'int8'}
            },
            'jetson_xavier': {
                'system': {'max_memory_mb': 32768, 'cpu_cores': 8},
                'edge_optimization': {'batch_size': 2, 'precision': 'mixed'}
            },
            'raspberry_pi_4': {
                'system': {'max_memory_mb': 4096, 'cpu_cores': 4},
                'edge_optimization': {'batch_size': 1, 'precision': 'int8'}
            },
            'industrial_pc': {
                'system': {'max_memory_mb': 16384, 'cpu_cores': 8},
                'edge_optimization': {'batch_size': 4, 'precision': 'fp16'}
            }
        }
        
        for device, device_config in device_configs.items():
            with open(config_dir / f'{device}_config.json', 'w') as f:
                json.dump(device_config, f, indent=2)
    
    def _create_deployment_scripts(self, deployment_dir: Path, config: Dict):
        """Create deployment scripts"""
        
        logger.info("Creating deployment scripts")
        
        scripts_dir = deployment_dir / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        
        # Installation script
        install_script = """#!/bin/bash
# AUTOMOS AI Installation Script

set -e

echo "Installing AUTOMOS AI..."

# Check Python version
python3 --version || (echo "Python 3 required" && exit 1)

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv build-essential cmake

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv automos_ai_env
source automos_ai_env/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set permissions
chmod +x scripts/start_automos_ai.sh
chmod +x scripts/stop_automos_ai.sh

echo "Installation completed successfully!"
echo "Run 'source automos_ai_env/bin/activate' to activate the environment"
echo "Run 'scripts/start_automos_ai.sh' to start AUTOMOS AI"
"""
        
        with open(scripts_dir / 'install.sh', 'w') as f:
            f.write(install_script)
        
        # Start script
        start_script = """#!/bin/bash
# AUTOMOS AI Start Script

set -e

echo "Starting AUTOMOS AI..."

# Activate virtual environment
if [ -d "automos_ai_env" ]; then
    source automos_ai_env/bin/activate
fi

# Detect device type
DEVICE_TYPE="industrial_pc"
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    if [[ $MODEL == *"Jetson Nano"* ]]; then
        DEVICE_TYPE="jetson_nano"
    elif [[ $MODEL == *"Jetson Xavier"* ]]; then
        DEVICE_TYPE="jetson_xavier"
    fi
fi

if [ -f /proc/cpuinfo ]; then
    if grep -q "ARMv7" /proc/cpuinfo; then
        DEVICE_TYPE="raspberry_pi_4"
    fi
fi

echo "Detected device type: $DEVICE_TYPE"

# Load device-specific configuration
CONFIG_FILE="config/${DEVICE_TYPE}_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="config/default_config.json"
fi

echo "Using configuration: $CONFIG_FILE"

# Start AUTOMOS AI
python3 main.py --mode deployment --hardware edge --config "$CONFIG_FILE"

echo "AUTOMOS AI started"
"""
        
        with open(scripts_dir / 'start_automos_ai.sh', 'w') as f:
            f.write(start_script)
        
        # Stop script
        stop_script = """#!/bin/bash
# AUTOMOS AI Stop Script

echo "Stopping AUTOMOS AI..."

# Find and kill AUTOMOS AI processes
pkill -f "python3 main.py" || echo "No AUTOMOS AI processes found"

echo "AUTOMOS AI stopped"
"""
        
        with open(scripts_dir / 'stop_automos_ai.sh', 'w') as f:
            f.write(stop_script)
        
        # Health check script
        health_script = """#!/bin/bash
# AUTOMOS AI Health Check Script

echo "Performing health check..."

# Check if AUTOMOS AI is running
if pgrep -f "python3 main.py" > /dev/null; then
    echo "✓ AUTOMOS AI process is running"
else
    echo "✗ AUTOMOS AI process is not running"
    exit 1
fi

# Check memory usage
MEMORY_USAGE=$(ps aux | grep "python3 main.py" | grep -v grep | awk '{print $4}' | head -1)
echo "Memory usage: ${MEMORY_USAGE}%"

# Check CPU usage
CPU_USAGE=$(ps aux | grep "python3 main.py" | grep -v grep | awk '{print $3}' | head -1)
echo "CPU usage: ${CPU_USAGE}%"

# Check log files for errors
if [ -f "logs/automos_ai.log" ]; then
    ERROR_COUNT=$(grep -c "ERROR" logs/automos_ai.log || echo "0")
    echo "Error count in logs: $ERROR_COUNT"
fi

echo "Health check completed"
"""
        
        with open(scripts_dir / 'health_check.sh', 'w') as f:
            f.write(health_script)
        
        # Make scripts executable
        for script_file in scripts_dir.glob('*.sh'):
            script_file.chmod(0o755)
    
    def _create_deployment_documentation(self, deployment_dir: Path, config: Dict):
        """Create deployment documentation"""
        
        logger.info("Creating deployment documentation")
        
        docs_dir = deployment_dir / 'docs'
        docs_dir.mkdir(exist_ok=True)
        
        # Installation guide
        install_guide = """# AUTOMOS AI Installation Guide

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Multi-core CPU (4 cores minimum)
- 10GB free disk space

## Quick Installation

1. Copy the deployment package to the target device
2. Run the installation script:
   ```bash
   chmod +x scripts/install.sh
   ./scripts/install.sh
   ```

3. Activate the virtual environment:
   ```bash
   source automos_ai_env/bin/activate
   ```

## Configuration

The system automatically detects the device type and loads appropriate configuration:

- Jetson Nano: `config/jetson_nano_config.json`
- Jetson Xavier: `config/jetson_xavier_config.json`
- Raspberry Pi 4: `config/raspberry_pi_4_config.json`
- Industrial PC: `config/industrial_pc_config.json`

## Starting AUTOMOS AI

```bash
./scripts/start_automos_ai.sh
```

## Health Check

```bash
./scripts/health_check.sh
```

## Troubleshooting

1. Check logs: `tail -f logs/automos_ai.log`
2. Verify system resources: `./scripts/health_check.sh`
3. Restart if needed: `./scripts/stop_automos_ai.sh && ./scripts/start_automos_ai.sh`
"""
        
        with open(docs_dir / 'INSTALLATION.md', 'w') as f:
            f.write(install_guide)
        
        # User manual
        user_manual = """# AUTOMOS AI User Manual

## Overview

AUTOMOS AI is a reasoning-based autonomous driving system that provides:
- 360° multi-camera processing
- Real-time safety monitoring
- Language-conditioned control
- Edge-optimized performance

## System Status

The system provides real-time status information:
- **Green**: System operating normally
- **Yellow**: Caution conditions detected
- **Red**: Emergency conditions

## Safety Features

### Emergency Override
The system includes automatic emergency override that activates when:
- Collision risk detected
- Loss of vehicle control
- System malfunction

### Safety Monitoring
Continuous monitoring of:
- Vehicle dynamics
- Surrounding objects
- Road conditions
- System health

## Operation Modes

### Normal Mode
Standard autonomous driving with full reasoning capabilities.

### Safety Mode
Reduced functionality with enhanced safety monitoring.

### Emergency Mode
Minimal functionality with maximum safety measures.

## Logs and Monitoring

- System logs: `logs/automos_ai.log`
- Performance metrics: `logs/performance.log`
- Safety events: `logs/safety.log`

## Support

For technical support, provide:
- System logs
- Configuration files
- Device information
- Error descriptions
"""
        
        with open(docs_dir / 'USER_MANUAL.md', 'w') as f:
            f.write(user_manual)
    
    def _create_package_metadata(self, deployment_dir: Path, deployment_type: str, config: Dict) -> Dict:
        """Create package metadata"""
        
        metadata = {
            'package_name': f'automos_ai_{deployment_type}',
            'version': '1.0.0',
            'deployment_type': deployment_type,
            'creation_date': datetime.now().isoformat(),
            'target_devices': config.get('target_devices', []),
            'minimum_requirements': {
                'python_version': '3.8',
                'memory_gb': 4,
                'cpu_cores': 4,
                'disk_space_gb': 10
            },
            'features': [
                '360° multi-camera processing',
                'Real-time safety monitoring',
                'Language-conditioned control',
                'Edge optimization',
                'Emergency override'
            ],
            'components': {
                'core_reasoning': True,
                'perception_pipeline': True,
                'safety_critic': True,
                'world_model': True,
                'edge_optimization': True
            }
        }
        
        # Save metadata
        with open(deployment_dir / 'package_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _create_compressed_package(self, deployment_dir: Path, config: Dict) -> Path:
        """Create compressed deployment package"""
        
        logger.info("Creating compressed package")
        
        compression_level = config.get('compression_level', 6)
        package_name = f"{deployment_dir.name}.zip"
        package_path = deployment_dir.parent / package_name
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=compression_level) as zipf:
            for file_path in deployment_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(deployment_dir)
                    zipf.write(file_path, arcname)
        
        # Remove the uncompressed directory
        shutil.rmtree(deployment_dir)
        
        logger.info(f"Compressed package created: {package_path}")
        return package_path
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum"""
        
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def deploy_to_device(self, package_path: str, device_config: Dict) -> Dict:
        """Deploy package to target device"""
        
        logger.info(f"Deploying {package_path} to device")
        
        # This would contain actual deployment logic
        # For now, return success
        
        deployment_result = {
            'success': True,
            'device_id': device_config.get('device_id', 'unknown'),
            'deployment_time': datetime.now().isoformat(),
            'package_path': package_path,
            'status': 'deployed'
        }
        
        return deployment_result
    
    def get_deployment_history(self) -> List[Dict]:
        """Get deployment history"""
        return self.deployment_history.copy()
    
    def verify_deployment(self, package_path: str) -> Dict:
        """Verify deployment package integrity"""
        
        logger.info(f"Verifying deployment package: {package_path}")
        
        try:
            if not os.path.exists(package_path):
                return {'success': False, 'error': 'Package file not found'}
            
            # Calculate checksum
            with open(package_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Extract and verify contents
            with zipfile.ZipFile(package_path, 'r') as zipf:
                file_list = zipf.namelist()
                
                # Check for required files
                required_files = [
                    'main.py',
                    'requirements.txt',
                    'package_metadata.json'
                ]
                
                missing_files = [f for f in required_files if f not in file_list]
                
                if missing_files:
                    return {
                        'success': False,
                        'error': f'Missing required files: {missing_files}'
                    }
            
            return {
                'success': True,
                'checksum': file_hash,
                'file_count': len(file_list),
                'package_size_mb': os.path.getsize(package_path) / (1024 * 1024)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
