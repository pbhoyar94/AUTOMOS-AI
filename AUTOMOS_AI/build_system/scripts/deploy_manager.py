#!/usr/bin/env python3
"""
AUTOMOS AI Deployment Manager
Complete deployment and installation management system
"""

import os
import sys
import json
import time
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class AUTOMOSDeployManager:
    """AUTOMOS AI Deployment Manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.deployments_dir = self.project_root / "deployments"
        self.templates_dir = self.project_root / "build_system" / "templates"
        
        # Ensure directories exist
        self.deployments_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
    
    def create_deployment(self, platform: str, target_os: str = "linux", 
                         features: Optional[List[str]] = None) -> Dict:
        """Create deployment package for specified platform"""
        
        print(f"📦 Creating deployment package for {platform} ({target_os})")
        
        # Platform configurations
        platform_configs = {
            'jetson_nano': {
                'arch': 'arm64',
                'memory': '4GB',
                'storage': '32GB',
                'default_features': ['basic_perception', 'safety_critic', 'edge_optimization']
            },
            'jetson_xavier': {
                'arch': 'arm64',
                'memory': '8GB',
                'storage': '64GB',
                'default_features': ['full_perception', 'safety_critic', 'ai_reasoning', 'edge_optimization']
            },
            'jetson_orin': {
                'arch': 'arm64',
                'memory': '16GB',
                'storage': '128GB',
                'default_features': ['full_perception', 'safety_critic', 'ai_reasoning', 'world_model', 'edge_optimization']
            },
            'drive_agx': {
                'arch': 'arm64',
                'memory': '32GB',
                'storage': '256GB',
                'default_features': ['all_features', 'automotive_safety', 'redundant_systems']
            },
            'industrial_pc': {
                'arch': 'x86_64',
                'memory': '16GB',
                'storage': '512GB',
                'default_features': ['full_perception', 'safety_critic', 'ai_reasoning', 'world_model']
            }
        }
        
        if platform not in platform_configs:
            raise ValueError(f"Unsupported platform: {platform}")
        
        config = platform_configs[platform]
        features = features or config['default_features']
        
        # Create deployment directory
        deployment_name = f"automos_ai_{platform}_{target_os}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        deployment_path = self.deployments_dir / deployment_name
        deployment_path.mkdir(exist_ok=True)
        
        print(f"📁 Deployment directory: {deployment_path}")
        
        # Create deployment structure
        self.create_deployment_structure(deployment_path, platform, target_os)
        
        # Copy and configure files
        self.copy_source_files(deployment_path, platform, target_os, features)
        
        # Create configuration files
        self.create_configuration_files(deployment_path, platform, target_os, features)
        
        # Create installation scripts
        self.create_installation_scripts(deployment_path, platform, target_os)
        
        # Create service files
        self.create_service_files(deployment_path, platform, target_os)
        
        # Create documentation
        self.create_documentation(deployment_path, platform, target_os)
        
        # Create deployment metadata
        metadata = self.create_deployment_metadata(deployment_path, platform, target_os, features)
        
        # Create deployment archive
        archive_path = self.create_deployment_archive(deployment_path)
        
        print(f"✅ Deployment package created: {archive_path}")
        
        return {
            'deployment_name': deployment_name,
            'deployment_path': str(deployment_path),
            'archive_path': str(archive_path),
            'platform': platform,
            'target_os': target_os,
            'features': features,
            'metadata': metadata
        }
    
    def create_deployment_structure(self, deployment_path: Path, platform: str, target_os: str):
        """Create deployment directory structure"""
        
        print("🏗️ Creating deployment structure...")
        
        # Create directories
        directories = [
            "bin",
            "lib",
            "config",
            "scripts",
            "data",
            "logs",
            "models",
            "docs",
            "tests",
            "boot",
            "rootfs"
        ]
        
        for directory in directories:
            (deployment_path / directory).mkdir(exist_ok=True)
        
        # Create platform-specific directories
        if platform in ['jetson_nano', 'jetson_xavier', 'jetson_orin']:
            (deployment_path / "jetson").mkdir(exist_ok=True)
        elif platform == 'drive_agx':
            (deployment_path / "drive").mkdir(exist_ok=True)
        elif platform == 'industrial_pc':
            (deployment_path / "industrial").mkdir(exist_ok=True)
        
        # Create OS-specific directories
        if target_os == 'qnx':
            (deployment_path / "qnx").mkdir(exist_ok=True)
        elif target_os == 'linux':
            (deployment_path / "linux").mkdir(exist_ok=True)
    
    def copy_source_files(self, deployment_path: Path, platform: str, target_os: str, features: List[str]):
        """Copy and configure source files"""
        
        print("📄 Copying source files...")
        
        # Feature to module mapping
        feature_modules = {
            'basic_perception': ['camera_processor', 'object_detector', 'lane_detector'],
            'full_perception': ['camera_processor', 'radar_processor', 'lidar_processor', 'sensor_fusion'],
            'safety_critic': ['safety_monitor', 'emergency_override', 'collision_avoidance'],
            'ai_reasoning': ['reasoning_engine', 'dualad_integration', 'opendrivevla_integration'],
            'world_model': ['predictive_mapping', 'hd_map_free_navigation'],
            'edge_optimization': ['model_quantizer', 'edge_optimizer', 'deployment_manager'],
            'automotive_safety': ['asil_d_monitoring', 'redundant_systems', 'fail_safe'],
            'redundant_systems': ['backup_systems', 'health_monitoring', 'auto_recovery'],
            'all_features': ['all_modules']
        }
        
        # Copy modules based on features
        for feature in features:
            modules = feature_modules.get(feature, [])
            
            for module in modules:
                if module == 'all_modules':
                    # Copy all modules
                    for src_dir in ['core', 'perception', 'safety', 'world_model', 'integration', 'edge_optimization']:
                        src_path = self.project_root / src_dir
                        if src_path.exists():
                            dst_path = deployment_path / "src" / src_dir
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                else:
                    # Copy specific module
                    src_path = self.project_root / module.replace('_', '/')
                    if src_path.exists():
                        dst_path = deployment_path / "src" / module
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        
        # Copy main application files
        main_files = ['main.py', 'requirements.txt', 'README.md']
        for file_name in main_files:
            src_file = self.project_root / file_name
            if src_file.exists():
                dst_file = deployment_path / file_name
                shutil.copy2(src_file, dst_file)
    
    def create_configuration_files(self, deployment_path: Path, platform: str, target_os: str, features: List[str]):
        """Create configuration files"""
        
        print("⚙️ Creating configuration files...")
        
        config_dir = deployment_path / "config"
        
        # Main configuration
        main_config = {
            'system': {
                'name': 'AUTOMOS AI',
                'version': '1.0.0',
                'platform': platform,
                'target_os': target_os,
                'build_date': datetime.now().isoformat()
            },
            'features': features,
            'hardware': {
                'platform': platform,
                'architecture': 'arm64' if 'jetson' in platform or platform == 'drive_agx' else 'x86_64',
                'memory': self.get_platform_memory(platform),
                'storage': self.get_platform_storage(platform)
            },
            'performance': {
                'target_fps': 30,
                'max_latency_ms': 10,
                'safety_response_ms': 10
            },
            'logging': {
                'level': 'INFO',
                'file': '/var/log/automos_ai/automos.log',
                'max_size': '100MB',
                'backup_count': 5
            }
        }
        
        with open(config_dir / "automos_config.json", 'w') as f:
            json.dump(main_config, f, indent=2)
        
        # Platform-specific configuration
        platform_config = self.create_platform_config(platform, target_os)
        with open(config_dir / "platform_config.json", 'w') as f:
            json.dump(platform_config, f, indent=2)
        
        # Feature-specific configurations
        for feature in features:
            feature_config = self.create_feature_config(feature, platform, target_os)
            with open(config_dir / f"{feature}_config.json", 'w') as f:
                json.dump(feature_config, f, indent=2)
    
    def create_installation_scripts(self, deployment_path: Path, platform: str, target_os: str):
        """Create installation scripts"""
        
        print("📜 Creating installation scripts...")
        
        scripts_dir = deployment_path / "scripts"
        
        # Main install script
        install_script = f"""#!/bin/bash
# AUTOMOS AI Installation Script
# Platform: {platform}
# Target OS: {target_os}

set -e

echo "Installing AUTOMOS AI for {platform} ({target_os})"

# Check permissions
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

# Create directories
mkdir -p /opt/automos_ai
mkdir -p /var/log/automos_ai
mkdir -p /etc/automos_ai
mkdir -p /usr/local/bin

# Copy files
echo "Copying application files..."
cp -r bin/* /opt/automos_ai/bin/
cp -r lib/* /opt/automos_ai/lib/
cp -r config/* /etc/automos_ai/
cp -r models/* /opt/automos_ai/models/

# Set permissions
chmod +x /opt/automos_ai/bin/*
chmod -R 755 /opt/automos_ai/
chmod -R 755 /etc/automos_ai/

# Create user
if ! id "automos" &>/dev/null; then
    useradd -r -s /bin/false automos
fi
chown -R automos:automos /opt/automos_ai/
chown -R automos:automos /var/log/automos_ai/

# Install dependencies
echo "Installing dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y python3 python3-pip opencv-python
elif command -v yum &> /dev/null; then
    yum install -y python3 python3-pip opencv-python
fi

# Install Python dependencies
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
fi

# Install service
if [ -f "scripts/automos_ai.service" ]; then
    cp scripts/automos_ai.service /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable automos_ai
fi

# Create startup script
cat > /usr/local/bin/automos_ai << 'EOF'
#!/bin/bash
/opt/automos_ai/bin/automos_ai "$@"
EOF
chmod +x /usr/local/bin/automos_ai

echo "Installation completed successfully!"
echo "Start AUTOMOS AI with: systemctl start automos_ai"
echo "Check status with: systemctl status automos_ai"
"""
        
        with open(scripts_dir / "install.sh", 'w') as f:
            f.write(install_script)
        (scripts_dir / "install.sh").chmod(0o755)
        
        # Uninstall script
        uninstall_script = """#!/bin/bash
# AUTOMOS AI Uninstallation Script

set -e

echo "Uninstalling AUTOMOS AI..."

# Stop service
systemctl stop automos_ai 2>/dev/null || true
systemctl disable automos_ai 2>/dev/null || true

# Remove files
rm -rf /opt/automos_ai
rm -rf /etc/automos_ai
rm -rf /var/log/automos_ai
rm -f /usr/local/bin/automos_ai
rm -f /etc/systemd/system/automos_ai.service

# Remove user
userdel automos 2>/dev/null || true

# Reload systemd
systemctl daemon-reload

echo "AUTOMOS AI uninstalled successfully!"
"""
        
        with open(scripts_dir / "uninstall.sh", 'w') as f:
            f.write(uninstall_script)
        (scripts_dir / "uninstall.sh").chmod(0o755)
        
        # Platform-specific scripts
        self.create_platform_scripts(scripts_dir, platform, target_os)
    
    def create_service_files(self, deployment_path: Path, platform: str, target_os: str):
        """Create service files"""
        
        print("🔧 Creating service files...")
        
        scripts_dir = deployment_path / "scripts"
        
        if target_os == 'linux':
            # Systemd service
            service_content = f"""[Unit]
Description=AUTOMOS AI Autonomous Driving System
After=network.target
Wants=network.target

[Service]
Type=simple
User=automos
Group=automos
WorkingDirectory=/opt/automos_ai
ExecStart=/opt/automos_ai/bin/automos_ai
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=automos_ai

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/automos_ai /opt/automos_ai/models

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
"""
            
            with open(scripts_dir / "automos_ai.service", 'w') as f:
                f.write(service_content)
        
        elif target_os == 'qnx':
            # QNX service script
            service_content = f"""#!/bin/sh
# AUTOMOS AI QNX Service Script

case "$1" in
    start)
        echo "Starting AUTOMOS AI..."
        /opt/automos_ai/bin/automos_ai &
        echo $! > /var/run/automos_ai.pid
        ;;
    stop)
        echo "Stopping AUTOMOS AI..."
        if [ -f /var/run/automos_ai.pid ]; then
            kill -TERM $(cat /var/run/automos_ai.pid)
            rm -f /var/run/automos_ai.pid
        fi
        ;;
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    *)
        echo "Usage: $0 {{start|stop|restart}}"
        exit 1
        ;;
esac
"""
            
            with open(scripts_dir / "automos_ai.sh", 'w') as f:
                f.write(service_content)
            (scripts_dir / "automos_ai.sh").chmod(0o755)
    
    def create_documentation(self, deployment_path: Path, platform: str, target_os: str):
        """Create documentation"""
        
        print("📚 Creating documentation...")
        
        docs_dir = deployment_path / "docs"
        
        # Installation guide
        install_guide = f"""# AUTOMOS AI Installation Guide

## Platform: {platform}
## Target OS: {target_os}
## Date: {datetime.now().strftime('%Y-%m-%d')}

### Prerequisites

1. {platform} hardware
2. {target_os} operating system
3. Sufficient storage ({self.get_platform_storage(platform)})
4. Network connectivity

### Installation Steps

1. Extract the deployment package:
   ```bash
   tar -xzf automos_ai_{platform}_{target_os}_*.tar.gz
   cd automos_ai_{platform}_{target_os}_*
   ```

2. Run installation script:
   ```bash
   sudo ./scripts/install.sh
   ```

3. Start the service:
   ```bash
   sudo systemctl start automos_ai
   ```

4. Verify installation:
   ```bash
   sudo systemctl status automos_ai
   ```

### Configuration

Configuration files are located in `/etc/automos_ai/`:
- `automos_config.json` - Main configuration
- `platform_config.json` - Platform-specific settings
- `*_config.json` - Feature-specific configurations

### Troubleshooting

#### Service fails to start
- Check logs: `journalctl -u automos_ai -f`
- Verify configuration files
- Check hardware compatibility

#### Performance issues
- Monitor system resources
- Adjust configuration parameters
- Check sensor connections

#### Network issues
- Verify network connectivity
- Check firewall settings
- Validate IP configurations

### Support

For support, contact: support@automos-ai.com
"""
        
        with open(docs_dir / "INSTALLATION.md", 'w') as f:
            f.write(install_guide)
        
        # User guide
        user_guide = f"""# AUTOMOS AI User Guide

## Platform: {platform}
## Target OS: {target_os}

### Overview

AUTOMOS AI is an autonomous driving system that provides:
- 360° perception using multiple sensors
- Real-time safety monitoring and emergency response
- AI-powered reasoning and decision making
- Edge-optimized performance

### Getting Started

1. Start AUTOMOS AI:
   ```bash
   sudo systemctl start automos_ai
   ```

2. Check status:
   ```bash
   sudo systemctl status automos_ai
   ```

3. View logs:
   ```bash
   sudo journalctl -u automos_ai -f
   ```

### Configuration

#### Main Configuration
Edit `/etc/automos_ai/automos_config.json` to modify:
- System settings
- Performance parameters
- Logging configuration

#### Platform Configuration
Edit `/etc/automos_ai/platform_config.json` to modify:
- Hardware-specific settings
- Sensor configurations
- Performance optimizations

### Features

#### Perception System
- Multi-camera processing (6 cameras)
- Radar and LiDAR integration
- Object detection and tracking
- Lane detection and marking

#### Safety System
- Real-time risk assessment
- Emergency override capability
- Collision avoidance
- <10ms response time

#### AI Reasoning
- LLM-powered decision making
- Natural language control
- Predictive modeling
- Adaptive behavior

#### Edge Optimization
- Model quantization
- Memory optimization
- Performance tuning
- Multi-device support

### Monitoring

#### System Status
```bash
automos_ai status
```

#### Performance Metrics
```bash
automos_ai metrics
```

#### Health Check
```bash
automos_ai health
```

### Maintenance

#### Updates
```bash
sudo systemctl stop automos_ai
# Update files
sudo systemctl start automos_ai
```

#### Backup Configuration
```bash
sudo cp -r /etc/automos_ai /backup/automos_ai_config_$(date +%Y%m%d)
```

### Safety

AUTOMOS AI includes multiple safety features:
- Continuous monitoring
- Emergency stop capability
- Fail-safe operation
- Redundant systems

Always ensure:
- Proper sensor calibration
- Regular system health checks
- Emergency procedures are in place
"""
        
        with open(docs_dir / "USER_GUIDE.md", 'w') as f:
            f.write(user_guide)
    
    def create_deployment_metadata(self, deployment_path: Path, platform: str, target_os: str, features: List[str]) -> Dict:
        """Create deployment metadata"""
        
        metadata = {
            'deployment_info': {
                'name': f"automos_ai_{platform}_{target_os}",
                'version': '1.0.0',
                'platform': platform,
                'target_os': target_os,
                'created': datetime.now().isoformat(),
                'features': features
            },
            'hardware_requirements': {
                'platform': platform,
                'architecture': 'arm64' if 'jetson' in platform or platform == 'drive_agx' else 'x86_64',
                'memory': self.get_platform_memory(platform),
                'storage': self.get_platform_storage(platform),
                'network': 'Gigabit Ethernet recommended'
            },
            'software_requirements': {
                'os': target_os,
                'python': '3.8+',
                'dependencies': [
                    'opencv-python',
                    'numpy',
                    'torch',
                    'pyyaml'
                ]
            },
            'installation': {
                'method': 'script',
                'requires_root': True,
                'estimated_time': '10-15 minutes',
                'post_install_steps': [
                    'Start service',
                    'Verify configuration',
                    'Check system status'
                ]
            },
            'features': {
                feature: self.get_feature_description(feature)
                for feature in features
            },
            'support': {
                'contact': 'support@automos-ai.com',
                'documentation': '/docs/',
                'logs': '/var/log/automos_ai/',
                'config': '/etc/automos_ai/'
            }
        }
        
        # Save metadata
        with open(deployment_path / "deployment_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def create_deployment_archive(self, deployment_path: Path) -> Path:
        """Create deployment archive"""
        
        print("📦 Creating deployment archive...")
        
        # Create archive
        archive_path = deployment_path.parent / f"{deployment_path.name}.tar.gz"
        
        # Use Python's tarfile for cross-platform compatibility
        import tarfile
        
        def filter_function(tarinfo):
            if '__pycache__' in tarinfo.name or tarinfo.name.endswith('.pyc'):
                return None
            return tarinfo
        
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(deployment_path, arcname=deployment_path.name, filter=filter_function)
        
        return archive_path
    
    def get_platform_memory(self, platform: str) -> str:
        """Get platform memory requirement"""
        memory_map = {
            'jetson_nano': '4GB',
            'jetson_xavier': '8GB',
            'jetson_orin': '16GB',
            'drive_agx': '32GB',
            'industrial_pc': '16GB'
        }
        return memory_map.get(platform, 'Unknown')
    
    def get_platform_storage(self, platform: str) -> str:
        """Get platform storage requirement"""
        storage_map = {
            'jetson_nano': '32GB',
            'jetson_xavier': '64GB',
            'jetson_orin': '128GB',
            'drive_agx': '256GB',
            'industrial_pc': '512GB'
        }
        return storage_map.get(platform, 'Unknown')
    
    def get_feature_description(self, feature: str) -> str:
        """Get feature description"""
        descriptions = {
            'basic_perception': 'Basic camera-based perception with object and lane detection',
            'full_perception': 'Complete perception pipeline with camera, radar, and LiDAR',
            'safety_critic': 'Real-time safety monitoring and emergency response system',
            'ai_reasoning': 'LLM-powered reasoning and decision making engine',
            'world_model': 'Predictive mapping and HD-map-free navigation',
            'edge_optimization': 'Model quantization and performance optimization',
            'automotive_safety': 'ASIL-D compliant automotive safety systems',
            'redundant_systems': 'Backup and recovery systems for reliability',
            'all_features': 'Complete AUTOMOS AI feature set'
        }
        return descriptions.get(feature, 'Unknown feature')
    
    def create_platform_config(self, platform: str, target_os: str) -> Dict:
        """Create platform-specific configuration"""
        
        base_config = {
            'platform': platform,
            'target_os': target_os,
            'architecture': 'arm64' if 'jetson' in platform or platform == 'drive_agx' else 'x86_64'
        }
        
        # Platform-specific settings
        if platform == 'jetson_nano':
            base_config.update({
                'gpu_memory': '4GB',
                'cpu_cores': 4,
                'max_power': '10W',
                'thermal_limit': '85°C'
            })
        elif platform == 'jetson_xavier':
            base_config.update({
                'gpu_memory': '8GB',
                'cpu_cores': 8,
                'max_power': '30W',
                'thermal_limit': '90°C'
            })
        elif platform == 'jetson_orin':
            base_config.update({
                'gpu_memory': '16GB',
                'cpu_cores': 12,
                'max_power': '50W',
                'thermal_limit': '95°C'
            })
        elif platform == 'drive_agx':
            base_config.update({
                'gpu_memory': '32GB',
                'cpu_cores': 16,
                'max_power': '100W',
                'thermal_limit': '105°C'
            })
        elif platform == 'industrial_pc':
            base_config.update({
                'gpu_memory': 'RTX 3060',
                'cpu_cores': 16,
                'max_power': '350W',
                'thermal_limit': '80°C'
            })
        
        return base_config
    
    def create_feature_config(self, feature: str, platform: str, target_os: str) -> Dict:
        """Create feature-specific configuration"""
        
        base_config = {
            'feature': feature,
            'enabled': True,
            'platform': platform,
            'target_os': target_os
        }
        
        # Feature-specific settings
        if 'perception' in feature:
            base_config.update({
                'camera_count': 6,
                'processing_fps': 30,
                'detection_threshold': 0.5,
                'tracking_enabled': True
            })
        elif 'safety' in feature:
            base_config.update({
                'response_time_ms': 10,
                'emergency_override': True,
                'collision_detection': True,
                'risk_assessment': True
            })
        elif 'reasoning' in feature:
            base_config.update({
                'llm_model': 'dualad_opendrivevla_fusion',
                'planning_horizon': 5.0,
                'decision_frequency': 10.0
            })
        elif 'optimization' in feature:
            base_config.update({
                'quantization': 'dynamic_int8',
                'memory_optimization': True,
                'performance_mode': 'balanced'
            })
        
        return base_config
    
    def create_platform_scripts(self, scripts_dir: Path, platform: str, target_os: str):
        """Create platform-specific scripts"""
        
        # Platform-specific startup script
        startup_script = f"""#!/bin/bash
# AUTOMOS AI Platform Startup Script
# Platform: {platform}

echo "Starting AUTOMOS AI on {platform}..."

# Platform-specific initialization
if [ "{platform}" = "jetson_nano" ]; then
    # Jetson Nano specific setup
    nvpmodel -m 0  # Max power mode
    jetson_clocks  # Max clocks
elif [ "{platform}" = "jetson_xavier" ]; then
    # Jetson Xavier specific setup
    nvpmodel -m 0  # Max power mode
    jetson_clocks  # Max clocks
elif [ "{platform}" = "jetson_orin" ]; then
    # Jetson Orin specific setup
    nvpmodel -m 0  # Max power mode
    jetson_clocks  # Max clocks
fi

# Start AUTOMOS AI
/opt/automos_ai/bin/automos_ai "$@"
"""
        
        with open(scripts_dir / "startup.sh", 'w') as f:
            f.write(startup_script)
        (scripts_dir / "startup.sh").chmod(0o755)
    
    def list_deployments(self):
        """List available deployment packages"""
        
        print("📦 Available Deployment Packages:")
        
        if not self.deployments_dir.exists():
            print("No deployment packages found")
            return
        
        for deployment in sorted(self.deployments_dir.glob("*.tar.gz")):
            print(f"  {deployment.name}")
            
            # Extract metadata if available
            metadata_file = deployment.with_suffix('.json')
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    deployment_info = metadata.get('deployment_info', {})
                    print(f"    Platform: {deployment_info.get('platform', 'Unknown')}")
                    print(f"    Target OS: {deployment_info.get('target_os', 'Unknown')}")
                    print(f"    Created: {deployment_info.get('created', 'Unknown')}")
                    print(f"    Features: {', '.join(deployment_info.get('features', []))}")
                except Exception:
                    print("    Metadata: Unable to read")
            print()
    
    def deploy_to_hardware(self, deployment_package: str, platform: str, method: str = 'auto'):
        """Deploy package to hardware"""
        
        print(f"🚀 Deploying {deployment_package} to {platform}")
        
        deployment_path = Path(deployment_package)
        if not deployment_path.exists():
            raise FileNotFoundError(f"Deployment package not found: {deployment_path}")
        
        # Extract deployment
        extract_dir = deployment_path.parent / deployment_path.stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        
        print("Extracting deployment package...")
        shutil.unpack_archive(deployment_path, extract_dir)
        
        # Run installation
        install_script = extract_dir / "scripts" / "install.sh"
        if install_script.exists():
            print("Running installation script...")
            result = subprocess.run(['sudo', 'bash', str(install_script)], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Installation failed: {result.stderr}")
            
            print("✅ Installation completed successfully!")
        else:
            raise FileNotFoundError("Installation script not found in deployment package")
        
        return {
            'deployment_package': str(deployment_path),
            'platform': platform,
            'status': 'success',
            'message': f"Successfully deployed to {platform}"
        }

def main():
    """Main deployment manager interface"""
    
    parser = argparse.ArgumentParser(description='AUTOMOS AI Deployment Manager')
    parser.add_argument('--create', action='store_true',
                       help='Create new deployment package')
    parser.add_argument('--platform', required=True,
                       choices=['jetson_nano', 'jetson_xavier', 'jetson_orin', 'drive_agx', 'industrial_pc'],
                       help='Target platform')
    parser.add_argument('--target-os', default='linux', choices=['linux', 'qnx'],
                       help='Target operating system')
    parser.add_argument('--features', nargs='+',
                       help='Features to include')
    parser.add_argument('--deploy', 
                       help='Deploy package to hardware')
    parser.add_argument('--list', action='store_true',
                       help='List available deployment packages')
    
    args = parser.parse_args()
    
    deploy_manager = AUTOMOSDeployManager()
    
    if args.list:
        deploy_manager.list_deployments()
        return
    
    if args.create:
        try:
            result = deploy_manager.create_deployment(
                platform=args.platform,
                target_os=args.target_os,
                features=args.features
            )
            
            print(f"\n🎉 Deployment package created successfully!")
            print(f"📦 Package: {result['archive_path']}")
            print(f"🎯 Platform: {result['platform']}")
            print(f"💻 Target OS: {result['target_os']}")
            print(f"⚡ Features: {', '.join(result['features'])}")
            
        except Exception as e:
            print(f"❌ Deployment creation failed: {e}")
            sys.exit(1)
    
    elif args.deploy:
        try:
            result = deploy_manager.deploy_to_hardware(
                deployment_package=args.deploy,
                platform=args.platform
            )
            
            print(f"\n🎉 Deployment completed successfully!")
            print(f"📦 Package: {result['deployment_package']}")
            print(f"🎯 Platform: {result['platform']}")
            print(f"✅ Status: {result['status']}")
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
