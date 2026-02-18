# AUTOMOS AI Build System

Complete build and deployment system for AUTOMOS AI autonomous driving platform.

## 🎯 Overview

The AUTOMOS AI Build System provides:
- **Cross-platform compilation** for multiple hardware platforms
- **Multi-feature builds** with selective component compilation
- **Automated deployment** with hardware flashing capabilities
- **Package management** for distribution and installation
- **Configuration management** for different deployment scenarios

## 🏗️ Supported Platforms

| Platform | Architecture | OS Support | Features | Flash Method |
|----------|-------------|------------|----------|--------------|
| **Jetson Nano** | ARM64 | Linux | Basic Perception | SD Card |
| **Jetson Xavier** | ARM64 | Linux, QNX | Full Perception + AI | USB Flash |
| **Jetson Orin** | ARM64 | Linux, QNX | All Features | USB Flash |
| **DRIVE AGX** | ARM64 | Linux, QNX | Automotive Grade | Network Flash |
| **Industrial PC** | x86_64 | Linux, QNX | Full System | USB Boot |

## ⚡ Features

### Build Features
- **basic_perception**: Camera-based perception with object/lane detection
- **full_perception**: Complete perception with camera, radar, LiDAR
- **safety_critic**: Real-time safety monitoring and emergency response
- **ai_reasoning**: LLM-powered reasoning and decision making
- **world_model**: Predictive mapping and HD-map-free navigation
- **edge_optimization**: Model quantization and performance optimization
- **automotive_safety**: ASIL-D compliant automotive safety systems
- **redundant_systems**: Backup and recovery systems for reliability

### Deployment Features
- **Automated installation** with dependency management
- **Service management** with systemd/QNX integration
- **Configuration management** with platform-specific settings
- **Health monitoring** and system status reporting
- **Update management** with rollback capabilities

## 🚀 Quick Start

### 1. Build for Platform

```bash
# Build AUTOMOS AI for Jetson Xavier
python build_system/build.py \
    --platform jetson_xavier \
    --target-os linux \
    --features full_perception safety_critic ai_reasoning \
    --clean
```

### 2. Create Deployment Package

```bash
# Create deployment package
python build_system/scripts/deploy_manager.py \
    --create \
    --platform jetson_xavier \
    --target-os linux \
    --features full_perception safety_critic ai_reasoning
```

### 3. Flash to Hardware

```bash
# Flash to Jetson Xavier
python build_system/scripts/flash_manager.py \
    --platform jetson_xavier \
    --deployment deployments/automos_ai_jetson_xavier_linux_20240218_143022.tar.gz
```

## 📁 Build System Structure

```
build_system/
├── build.py              # Main build system
├── cmake/
│   └── CMakeLists.txt    # CMake configuration
├── scripts/
│   ├── build_manager.py   # Build management
│   ├── deploy_manager.py # Deployment management
│   └── flash_manager.py  # Flash management
├── configs/
│   ├── jetson_nano.json  # Platform configs
│   ├── jetson_xavier.json
│   └── ...
└── templates/
    ├── service.template  # Service templates
    ├── config.template   # Config templates
    └── install.template  # Install templates
```

## 🔧 Build Commands

### Build System

```bash
# List supported platforms
python build_system/build.py --list-platforms

# Build with default features
python build_system/build.py --platform jetson_xavier

# Build with specific features
python build_system/build.py \
    --platform jetson_xavier \
    --features full_perception safety_critic

# Build for QNX
python build_system/build.py \
    --platform jetson_xavier \
    --target-os qnx

# Clean build
python build_system/build.py \
    --platform jetson_xavier \
    --clean
```

### Deployment Management

```bash
# List available deployments
python build_system/scripts/deploy_manager.py --list

# Create deployment package
python build_system/scripts/deploy_manager.py \
    --create \
    --platform jetson_xavier \
    --target-os linux

# Deploy to hardware
python build_system/scripts/deploy_manager.py \
    --deploy automos_ai_jetson_xavier_linux.tar.gz \
    --platform jetson_xavier
```

### Flash Management

```bash
# Flash to hardware
python build_system/scripts/flash_manager.py \
    --platform jetson_xavier \
    --deployment deployments/automos_ai_jetson_xavier_linux.tar.gz

# Flash with specific method
python build_system/scripts/flash_manager.py \
    --platform jetson_xavier \
    --deployment deployments/automos_ai_jetson_xavier_linux.tar.gz \
    --method usb_flash
```

## 📦 Deployment Packages

### Package Structure

```
automos_ai_jetson_xavier_linux_20240218_143022/
├── bin/                   # Executables
├── lib/                   # Libraries
├── config/                # Configuration files
├── scripts/               # Installation scripts
├── data/                  # Data files
├── models/                # AI models
├── docs/                  # Documentation
├── tests/                 # Test files
├── boot/                  # Boot files
├── rootfs/                # Root filesystem
├── deployment_metadata.json
└── FLASHING_INSTRUCTIONS.md
```

### Installation

```bash
# Extract package
tar -xzf automos_ai_jetson_xavier_linux_20240218_143022.tar.gz
cd automos_ai_jetson_xavier_linux_20240218_143022

# Install
sudo ./scripts/install.sh

# Start service
sudo systemctl start automos_ai

# Check status
sudo systemctl status automos_ai
```

## 🔧 Configuration

### Main Configuration

```json
{
  "system": {
    "name": "AUTOMOS AI",
    "version": "1.0.0",
    "platform": "jetson_xavier",
    "target_os": "linux"
  },
  "features": ["full_perception", "safety_critic", "ai_reasoning"],
  "hardware": {
    "platform": "jetson_xavier",
    "architecture": "arm64",
    "memory": "8GB",
    "storage": "64GB"
  },
  "performance": {
    "target_fps": 30,
    "max_latency_ms": 10,
    "safety_response_ms": 10
  }
}
```

### Platform Configuration

```json
{
  "platform": "jetson_xavier",
  "target_os": "linux",
  "architecture": "arm64",
  "gpu_memory": "8GB",
  "cpu_cores": 8,
  "max_power": "30W",
  "thermal_limit": "90°C"
}
```

## 🚨 Flash Methods

### SD Card Flash (Jetson Nano)

```bash
# Automatic detection and flashing
python build_system/scripts/flash_manager.py \
    --platform jetson_nano \
    --deployment deployments/automos_ai_jetson_nano_linux.tar.gz \
    --method sd_card
```

### USB Flash (Jetson Xavier/Orin)

```bash
# Requires device in recovery mode
python build_system/scripts/flash_manager.py \
    --platform jetson_xavier \
    --deployment deployments/automos_ai_jetson_xavier_linux.tar.gz \
    --method usb_flash
```

### Network Flash (DRIVE AGX)

```bash
# Requires network connection
python build_system/scripts/flash_manager.py \
    --platform drive_agx \
    --deployment deployments/automos_ai_drive_agx_linux.tar.gz \
    --method network_flash
```

## 📊 Performance Optimization

### Build Optimization

```bash
# Release build with optimizations
python build_system/build.py \
    --platform jetson_xavier \
    --target-os linux \
    --clean

# Enable CUDA acceleration
export CUDA_ENABLED=true
python build_system/build.py \
    --platform jetson_xavier \
    --target-os linux
```

### Runtime Optimization

```json
{
  "edge_optimization": {
    "quantization": "dynamic_int8",
    "memory_optimization": true,
    "performance_mode": "balanced",
    "gpu_acceleration": true
  }
}
```

## 🛡️ Safety and Certification

### ASIL-D Compliance

```json
{
  "automotive_safety": {
    "asil_level": "D",
    "redundant_systems": true,
    "fail_safe_operation": true,
    "emergency_response_ms": 10,
    "health_monitoring": true
  }
}
```

### Safety Features

- **Real-time monitoring** with <10ms response
- **Emergency override** capability
- **Redundant systems** for reliability
- **Fail-safe operation** with graceful degradation
- **Health monitoring** and auto-recovery

## 🧪 Testing

### Build Testing

```bash
# Run build tests
python build_system/scripts/test_build.py --platform jetson_xavier

# Run integration tests
python build_system/scripts/test_integration.py --platform jetson_xavier
```

### Deployment Testing

```bash
# Test deployment package
python build_system/scripts/test_deployment.py \
    --deployment deployments/automos_ai_jetson_xavier_linux.tar.gz

# Test installation
python build_system/scripts/test_installation.py --platform jetson_xavier
```

## 🔍 Troubleshooting

### Build Issues

```bash
# Clean build
python build_system/build.py --platform jetson_xavier --clean

# Check dependencies
python build_system/scripts/check_deps.py --platform jetson_xavier

# Verify toolchain
python build_system/scripts/verify_toolchain.py --platform jetson_xavier
```

### Flash Issues

```bash
# Check device connection
python build_system/scripts/check_device.py --platform jetson_xavier

# Verify flash method
python build_system/scripts/verify_flash.py --platform jetson_xavier

# Recovery mode
python build_system/scripts/recovery_flash.py --platform jetson_xavier
```

### Runtime Issues

```bash
# Check system status
automos_ai status

# View logs
sudo journalctl -u automos_ai -f

# Health check
automos_ai health
```

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## 🆘 Support

For support:
- **Email**: support@automos-ai.com
- **Documentation**: /docs/
- **Logs**: /var/log/automos_ai/
- **Configuration**: /etc/automos_ai/

## 📄 License

AUTOMOS AI Build System - Copyright 2024 AUTOMOS AI

All rights reserved.
