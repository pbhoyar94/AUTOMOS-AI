#!/usr/bin/env python3
"""
AUTOMOS AI Deployment Package Creator
Creates production deployment packages without external dependencies
"""

import os
import sys
import zipfile
import json
import hashlib
from pathlib import Path
from datetime import datetime

def create_deployment_package():
    """Create deployment package for customer deployment"""
    
    print("Creating AUTOMOS AI Production Deployment Package")
    
    # Get current directory
    current_dir = Path(__file__).parent
    output_dir = current_dir / "deployments"
    output_dir.mkdir(exist_ok=True)
    
    # Create deployment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    deployment_dir = output_dir / f"automos_ai_deployment_production_{timestamp}"
    deployment_dir.mkdir(exist_ok=True)
    
    print(f"Creating deployment package in: {deployment_dir}")
    
    # Copy core directories
    core_dirs = [
        'core', 'perception', 'safety', 'world_model', 
        'integration', 'edge_optimization', 'monitoring'
    ]
    
    for dir_name in core_dirs:
        src_dir = current_dir / dir_name
        dst_dir = deployment_dir / dir_name
        
        if src_dir.exists():
            print(f"Copying {dir_name}...")
            copy_directory(src_dir, dst_dir)
    
    # Copy core files
    core_files = ['main.py', 'requirements.txt', 'README.md', 'CUSTOMER_DEPLOYMENT.md']
    
    for file_name in core_files:
        src_file = current_dir / file_name
        dst_file = deployment_dir / file_name
        
        if src_file.exists():
            print(f"Copying {file_name}...")
            import shutil
            shutil.copy2(src_file, dst_file)
    
    # Create scripts directory
    scripts_dir = deployment_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    # Create installation script
    install_script = """#!/bin/bash
# AUTOMOS AI Installation Script

set -e

echo "Installing AUTOMOS AI..."

# Check Python version
python3 --version || (echo "Python 3 required" && exit 1)

# Install system dependencies
echo "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv build-essential cmake
elif command -v yum &> /dev/null; then
    sudo yum update -y
    sudo yum install -y python3-pip python3-venv gcc gcc-c++ make cmake
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv automos_ai_env
source automos_ai_env/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy scipy
pip install psutil pyyaml
pip install -r requirements.txt

# Set permissions
chmod +x scripts/start_automos_ai.sh
chmod +x scripts/stop_automos_ai.sh
chmod +x scripts/health_check.sh

echo "✅ Installation completed successfully!"
echo "🎯 Run 'source automos_ai_env/bin/activate' to activate the environment"
echo "🚀 Run 'scripts/start_automos_ai.sh' to start AUTOMOS AI"
"""
    
    with open(scripts_dir / "install.sh", "w") as f:
        f.write(install_script)
    
    # Create start script
    start_script = """#!/bin/bash
# AUTOMOS AI Start Script

set -e

echo "🚀 Starting AUTOMOS AI..."

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

echo "🔍 Detected device type: $DEVICE_TYPE"

# Create logs directory
mkdir -p logs

# Start AUTOMOS AI
echo "🧠 Starting AUTOMOS AI with $DEVICE_TYPE configuration..."
python3 main.py --mode deployment --hardware edge 2>&1 | tee logs/automos_ai.log

echo "✅ AUTOMOS AI started"
"""
    
    with open(scripts_dir / "start_automos_ai.sh", "w") as f:
        f.write(start_script)
    
    # Create health check script
    health_script = """#!/bin/bash
# AUTOMOS AI Health Check Script

echo "🏥 Performing health check..."

# Check if AUTOMOS AI is running
if pgrep -f "python3 main.py" > /dev/null; then
    echo "✅ AUTOMOS AI process is running"
else
    echo "❌ AUTOMOS AI process is not running"
    exit 1
fi

# Check memory usage
if command -v ps &> /dev/null; then
    MEMORY_USAGE=$(ps aux | grep "python3 main.py" | grep -v grep | awk '{print $4}' | head -1)
    echo "💾 Memory usage: ${MEMORY_USAGE}%"
fi

# Check CPU usage
if command -v ps &> /dev/null; then
    CPU_USAGE=$(ps aux | grep "python3 main.py" | grep -v grep | awk '{print $3}' | head -1)
    echo "⚡ CPU usage: ${CPU_USAGE}%"
fi

# Check log files for errors
if [ -f "logs/automos_ai.log" ]; then
    ERROR_COUNT=$(grep -c "ERROR" logs/automos_ai.log 2>/dev/null || echo "0")
    echo "📊 Error count in logs: $ERROR_COUNT"
    
    WARNING_COUNT=$(grep -c "WARNING" logs/automos_ai.log 2>/dev/null || echo "0")
    echo "⚠️  Warning count in logs: $WARNING_COUNT"
fi

echo "✅ Health check completed"
"""
    
    with open(scripts_dir / "health_check.sh", "w") as f:
        f.write(health_script)
    
    # Create stop script
    stop_script = """#!/bin/bash
# AUTOMOS AI Stop Script

echo "🛑 Stopping AUTOMOS AI..."

# Find and kill AUTOMOS AI processes
if pgrep -f "python3 main.py" > /dev/null; then
    pkill -f "python3 main.py"
    echo "✅ AUTOMOS AI processes stopped"
else
    echo "ℹ️  No AUTOMOS AI processes found"
fi

echo "🛑 AUTOMOS AI stopped"
"""
    
    with open(scripts_dir / "stop_automos_ai.sh", "w") as f:
        f.write(stop_script)
    
    # Make scripts executable
    for script_file in scripts_dir.glob("*.sh"):
        os.chmod(script_file, 0o755)
    
    # Create config directory
    config_dir = deployment_dir / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Create default configuration
    default_config = {
        "system": {
            "mode": "deployment",
            "hardware": "edge",
            "log_level": "INFO",
            "max_memory_mb": 4096,
            "cpu_cores": 4
        },
        "perception": {
            "camera_count": 6,
            "radar_count": 4,
            "lidar_count": 2,
            "processing_frequency": 20
        },
        "safety": {
            "response_time_ms": 10,
            "emergency_override": True,
            "max_acceleration": 3.0,
            "max_deceleration": 8.0
        },
        "reasoning": {
            "llm_model": "dualad_opendrivevla_fusion",
            "planning_algorithm": "lattice_idm_vla"
        },
        "edge_optimization": {
            "quantization_type": "dynamic_int8",
            "batch_size": 1,
            "precision": "int8"
        }
    }
    
    with open(config_dir / "default_config.json", "w") as f:
        json.dump(default_config, f, indent=2)
    
    # Create device-specific configurations
    device_configs = {
        "jetson_nano": {
            "system": {"max_memory_mb": 4096, "cpu_cores": 4},
            "edge_optimization": {"batch_size": 1, "precision": "int8"}
        },
        "jetson_xavier": {
            "system": {"max_memory_mb": 32768, "cpu_cores": 8},
            "edge_optimization": {"batch_size": 2, "precision": "mixed"}
        },
        "raspberry_pi_4": {
            "system": {"max_memory_mb": 4096, "cpu_cores": 4},
            "edge_optimization": {"batch_size": 1, "precision": "int8"}
        },
        "industrial_pc": {
            "system": {"max_memory_mb": 16384, "cpu_cores": 8},
            "edge_optimization": {"batch_size": 4, "precision": "fp16"}
        }
    }
    
    for device, config in device_configs.items():
        with open(config_dir / f"{device}_config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    # Create models directory
    models_dir = deployment_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Create model info
    model_info = {
        "models": {
            "reasoning_engine": "reasoning_engine_dynamic_int8.pt",
            "perception_pipeline": "perception_pipeline_dynamic_int8.pt",
            "safety_critic": "safety_critic_dynamic_int8.pt",
            "world_model": "world_model_dynamic_int8.pt"
        },
        "quantization_type": "dynamic_int8",
        "target_devices": ["jetson_nano", "jetson_xavier", "raspberry_pi_4", "industrial_pc"],
        "note": "Models will be downloaded and quantized during first startup"
    }
    
    with open(models_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Create package metadata
    package_metadata = {
        "package_name": "automos_ai_production",
        "version": "1.0.0",
        "deployment_type": "production",
        "creation_date": datetime.now().isoformat(),
        "target_devices": ["jetson_nano", "jetson_xavier", "raspberry_pi_4", "industrial_pc"],
        "minimum_requirements": {
            "python_version": "3.8",
            "memory_gb": 4,
            "cpu_cores": 4,
            "disk_space_gb": 10
        },
        "features": [
            "360° multi-camera processing",
            "Real-time safety monitoring",
            "Language-conditioned control",
            "Edge optimization",
            "Emergency override",
            "HD-map-free navigation",
            "Social intent recognition"
        ],
        "components": {
            "core_reasoning": True,
            "perception_pipeline": True,
            "safety_critic": True,
            "world_model": True,
            "edge_optimization": True,
            "monitoring_system": True
        }
    }
    
    with open(deployment_dir / "package_metadata.json", "w") as f:
        json.dump(package_metadata, f, indent=2)
    
    # Create compressed package
    package_name = f"automos_ai_deployment_production_{timestamp}.zip"
    package_path = output_dir / package_name
    
    print(f"📦 Creating compressed package: {package_path}")
    
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for file_path in deployment_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(deployment_dir)
                zipf.write(file_path, arcname)
    
    # Calculate checksum
    print("🔐 Calculating checksum...")
    sha256_hash = hashlib.sha256()
    with open(package_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    checksum = sha256_hash.hexdigest()
    
    # Get package size
    package_size_mb = os.path.getsize(package_path) / (1024 * 1024)
    
    # Remove uncompressed directory
    import shutil
    shutil.rmtree(deployment_dir)
    
    # Create deployment info
    deployment_info = {
        "success": True,
        "package_path": str(package_path),
        "package_name": package_name,
        "package_size_mb": round(package_size_mb, 2),
        "checksum": checksum,
        "creation_time": datetime.now().isoformat(),
        "features": package_metadata["features"],
        "target_devices": package_metadata["target_devices"]
    }
    
    # Save deployment info
    with open(output_dir / "deployment_info.json", "w") as f:
        json.dump(deployment_info, f, indent=2)
    
    print("✅ Production deployment package created successfully!")
    print(f"📦 Package: {package_path}")
    print(f"📏 Size: {package_size_mb:.1f} MB")
    print(f"🔐 Checksum: {checksum}")
    print(f"🎯 Target devices: {', '.join(package_metadata['target_devices'])}")
    print(f"⚡ Features: {len(package_metadata['features'])} advanced features")
    
    return deployment_info

def copy_directory(src, dst):
    """Copy directory without __pycache__ files"""
    import shutil
    
    if dst.exists():
        shutil.rmtree(dst)
    
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))

if __name__ == "__main__":
    try:
        result = create_deployment_package()
        if result["success"]:
            print("\n🎉 AUTOMOS AI is ready for customer deployment!")
            print("📖 See CUSTOMER_DEPLOYMENT.md for detailed instructions")
            sys.exit(0)
        else:
            print("❌ Failed to create deployment package")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error creating deployment package: {e}")
        sys.exit(1)
