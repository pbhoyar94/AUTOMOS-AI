#!/usr/bin/env python3
"""
AUTOMOS AI Build System
Cross-platform build and flash system for multiple hardware platforms
"""

import os
import sys
import json
import subprocess
import shutil
import argparse
import time
from pathlib import Path
from datetime import datetime

class AUTOMOSBuildSystem:
    """AUTOMOS AI Build System for multiple platforms"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.build_dir = self.project_root / "build"
        self.config_dir = self.project_root / "build_system" / "configs"
        
        # Platform configurations
        self.platforms = {
            'jetson_nano': {
                'arch': 'arm64',
                'toolchain': 'aarch64-linux-gnu',
                'qnx_support': False,
                'features': ['basic_perception', 'safety_critic', 'edge_optimization'],
                'memory_limit': '4GB',
                'flash_method': 'sd_card'
            },
            'jetson_xavier': {
                'arch': 'arm64',
                'toolchain': 'aarch64-linux-gnu',
                'qnx_support': True,
                'features': ['full_perception', 'safety_critic', 'ai_reasoning', 'edge_optimization'],
                'memory_limit': '8GB',
                'flash_method': 'usb_flash'
            },
            'jetson_orin': {
                'arch': 'arm64',
                'toolchain': 'aarch64-linux-gnu',
                'qnx_support': True,
                'features': ['full_perception', 'safety_critic', 'ai_reasoning', 'world_model', 'edge_optimization'],
                'memory_limit': '16GB',
                'flash_method': 'usb_flash'
            },
            'drive_agx': {
                'arch': 'arm64',
                'toolchain': 'aarch64-linux-gnu',
                'qnx_support': True,
                'features': ['all_features', 'automotive_safety', 'redundant_systems'],
                'memory_limit': '32GB',
                'flash_method': 'network_flash'
            },
            'industrial_pc': {
                'arch': 'x86_64',
                'toolchain': 'x86_64-linux-gnu',
                'qnx_support': True,
                'features': ['full_perception', 'safety_critic', 'ai_reasoning', 'world_model'],
                'memory_limit': '16GB',
                'flash_method': 'usb_boot'
            }
        }
        
        # Feature configurations
        self.features = {
            'basic_perception': {
                'modules': ['camera_processor', 'object_detector', 'lane_detector'],
                'memory_req': '512MB',
                'cpu_req': '2 cores',
                'build_time': '5min'
            },
            'full_perception': {
                'modules': ['camera_processor', 'radar_processor', 'lidar_processor', 'sensor_fusion'],
                'memory_req': '2GB',
                'cpu_req': '4 cores',
                'build_time': '15min'
            },
            'safety_critic': {
                'modules': ['safety_monitor', 'emergency_override', 'collision_avoidance'],
                'memory_req': '256MB',
                'cpu_req': '1 core',
                'build_time': '3min'
            },
            'ai_reasoning': {
                'modules': ['reasoning_engine', 'dualad_integration', 'opendrivevla_integration'],
                'memory_req': '4GB',
                'cpu_req': '6 cores',
                'build_time': '30min'
            },
            'world_model': {
                'modules': ['predictive_mapping', 'hd_map_free_navigation'],
                'memory_req': '1GB',
                'cpu_req': '2 cores',
                'build_time': '10min'
            },
            'edge_optimization': {
                'modules': ['model_quantizer', 'edge_optimizer', 'deployment_manager'],
                'memory_req': '512MB',
                'cpu_req': '2 cores',
                'build_time': '8min'
            },
            'automotive_safety': {
                'modules': ['asil_d_monitoring', 'redundant_systems', 'fail_safe'],
                'memory_req': '1GB',
                'cpu_req': '4 cores',
                'build_time': '20min'
            },
            'redundant_systems': {
                'modules': ['backup_systems', 'health_monitoring', 'auto_recovery'],
                'memory_req': '2GB',
                'cpu_req': '4 cores',
                'build_time': '12min'
            },
            'all_features': {
                'modules': ['all_modules'],
                'memory_req': '8GB',
                'cpu_req': '8 cores',
                'build_time': '60min'
            }
        }
    
    def build(self, platform, features=None, target_os='linux', clean=False):
        """Build AUTOMOS AI for specified platform"""
        
        print(f"🔨 Building AUTOMOS AI for {platform} ({target_os})")
        
        if platform not in self.platforms:
            raise ValueError(f"Unsupported platform: {platform}")
        
        platform_config = self.platforms[platform]
        
        # Determine features to build
        if features is None:
            features = platform_config['features']
        
        # Clean build if requested
        if clean:
            self.clean_build(platform)
        
        # Create build directory
        build_path = self.build_dir / f"{platform}_{target_os}"
        build_path.mkdir(parents=True, exist_ok=True)
        
        # Build configuration
        build_config = {
            'platform': platform,
            'target_os': target_os,
            'arch': platform_config['arch'],
            'toolchain': platform_config['toolchain'],
            'features': features,
            'build_time': datetime.now().isoformat()
        }
        
        # Save build config
        with open(build_path / "build_config.json", 'w') as f:
            json.dump(build_config, f, indent=2)
        
        # Build each feature
        build_results = {}
        total_time = 0
        
        for feature in features:
            print(f"\n🔧 Building feature: {feature}")
            
            feature_config = self.features.get(feature, self.features['all_features'])
            
            # Build feature
            result = self.build_feature(feature, platform, target_os, build_path)
            build_results[feature] = result
            total_time += result.get('build_time', 0)
            
            if result['status'] == 'success':
                print(f"✅ {feature}: Built successfully ({result['build_time']:.1f}s)")
            else:
                print(f"❌ {feature}: Build failed - {result.get('error', 'Unknown error')}")
        
        # Create deployment package
        package_result = self.create_deployment_package(platform, target_os, build_path)
        
        # Build summary
        summary = {
            'platform': platform,
            'target_os': target_os,
            'features': features,
            'build_results': build_results,
            'package_result': package_result,
            'total_build_time': total_time,
            'status': 'success' if all(r['status'] == 'success' for r in build_results.values()) else 'partial'
        }
        
        # Save build summary
        with open(build_path / "build_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Build completed for {platform}")
        print(f"📊 Features built: {len([r for r in build_results.values() if r['status'] == 'success'])}/{len(features)}")
        print(f"⏱️ Total time: {total_time:.1f}s")
        print(f"📦 Package: {package_result['package_path']}")
        
        return summary
    
    def build_feature(self, feature, platform, target_os, build_path):
        """Build individual feature"""
        
        feature_config = self.features.get(feature, {})
        modules = feature_config.get('modules', [])
        
        feature_build_path = build_path / feature
        feature_build_path.mkdir(exist_ok=True)
        
        # Build modules
        module_results = {}
        
        for module in modules:
            if module == 'all_modules':
                # Build all modules
                result = self.build_all_modules(platform, target_os, feature_build_path)
            else:
                # Build individual module
                result = self.build_module(module, platform, target_os, feature_build_path)
            
            module_results[module] = result
        
        # Feature build result
        success_count = len([r for r in module_results.values() if r['status'] == 'success'])
        
        return {
            'feature': feature,
            'modules': module_results,
            'modules_built': success_count,
            'total_modules': len(modules),
            'status': 'success' if success_count == len(modules) else 'partial',
            'build_time': sum(r.get('build_time', 0) for r in module_results.values())
        }
    
    def build_module(self, module, platform, target_os, build_path):
        """Build individual module"""
        
        start_time = time.time()
        
        try:
            # Create module build directory
            module_path = build_path / module
            module_path.mkdir(exist_ok=True)
            
            # Copy source files
            source_path = self.project_root / module.replace('_', '/')
            if source_path.exists():
                shutil.copytree(source_path, module_path / "src", dirs_exist_ok=True)
            
            # Create build script
            build_script = self.create_module_build_script(module, platform, target_os, module_path)
            
            # Execute build
            result = subprocess.run(
                ['bash', str(build_script)],
                cwd=module_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            build_time = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    'module': module,
                    'status': 'success',
                    'build_time': build_time,
                    'output': result.stdout,
                    'artifacts': list(module_path.glob("*.so")) + list(module_path.glob("*.a"))
                }
            else:
                return {
                    'module': module,
                    'status': 'failed',
                    'build_time': build_time,
                    'error': result.stderr,
                    'output': result.stdout
                }
        
        except Exception as e:
            return {
                'module': module,
                'status': 'failed',
                'build_time': time.time() - start_time,
                'error': str(e)
            }
    
    def build_all_modules(self, platform, target_os, build_path):
        """Build all AUTOMOS AI modules"""
        
        start_time = time.time()
        
        try:
            # Create comprehensive build
            all_path = build_path / "automos_ai_complete"
            all_path.mkdir(exist_ok=True)
            
            # Copy all source code
            for src_dir in ['core', 'perception', 'safety', 'world_model', 'integration', 'edge_optimization']:
                src_path = self.project_root / src_dir
                if src_path.exists():
                    shutil.copytree(src_path, all_path / src_dir, dirs_exist_ok=True)
            
            # Create main build script
            build_script = self.create_main_build_script(platform, target_os, all_path)
            
            # Execute build
            result = subprocess.run(
                ['bash', str(build_script)],
                cwd=all_path,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            build_time = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    'module': 'all_modules',
                    'status': 'success',
                    'build_time': build_time,
                    'output': result.stdout,
                    'artifacts': list(all_path.glob("*.so")) + list(all_path.glob("*.a"))
                }
            else:
                return {
                    'module': 'all_modules',
                    'status': 'failed',
                    'build_time': build_time,
                    'error': result.stderr,
                    'output': result.stdout
                }
        
        except Exception as e:
            return {
                'module': 'all_modules',
                'status': 'failed',
                'build_time': time.time() - start_time,
                'error': str(e)
            }
    
    def create_module_build_script(self, module, platform, target_os, build_path):
        """Create build script for module"""
        
        platform_config = self.platforms[platform]
        arch = platform_config['arch']
        toolchain = platform_config['toolchain']
        
        script_content = f"""#!/bin/bash
# AUTOMOS AI Module Build Script
# Module: {module}
# Platform: {platform}
# Target OS: {target_os}

set -e

echo "Building {module} for {platform} ({target_os})"

# Setup cross-compilation environment
if [ "{arch}" = "arm64" ]; then
    export CC={toolchain}-gcc
    export CXX={toolchain}-g++
    export AR={toolchain}-ar
    export STRIP={toolchain}-strip
    export CROSS_COMPILE={toolchain}-
fi

# Build flags
CFLAGS="-O3 -fPIC -Wall -Wextra"
CXXFLAGS="$CFLAGS -std=c++17"
LDFLAGS="-shared"

if [ "{target_os}" = "qnx" ]; then
    CFLAGS="$CFLAGS -D_QNX_SOURCE"
    LDFLAGS="$LDFLAGS -lrt"
fi

# Create build directory
mkdir -p build
cd build

# Configure
cmake .. \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_C_COMPILER="$CC" \\
    -DCMAKE_CXX_COMPILER="$CXX" \\
    -DCMAKE_C_FLAGS="$CFLAGS" \\
    -DCMAKE_CXX_FLAGS="$CXXFLAGS" \\
    -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \\
    -DTARGET_PLATFORM="{platform}" \\
    -DTARGET_OS="{target_os}"

# Build
make -j$(nproc)

# Install
make install

echo "Build completed successfully"
"""
        
        script_path = build_path / "build.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        script_path.chmod(0o755)
        return script_path
    
    def create_main_build_script(self, platform, target_os, build_path):
        """Create main build script for all modules"""
        
        platform_config = self.platforms[platform]
        arch = platform_config['arch']
        toolchain = platform_config['toolchain']
        
        script_content = f"""#!/bin/bash
# AUTOMOS AI Main Build Script
# Platform: {platform}
# Target OS: {target_os}

set -e

echo "Building AUTOMOS AI for {platform} ({target_os})"

# Setup cross-compilation environment
if [ "{arch}" = "arm64" ]; then
    export CC={toolchain}-gcc
    export CXX={toolchain}-g++
    export AR={toolchain}-ar
    export STRIP={toolchain}-strip
    export CROSS_COMPILE={toolchain}-
fi

# Build flags
CFLAGS="-O3 -fPIC -Wall -Wextra"
CXXFLAGS="$CFLAGS -std=c++17"
LDFLAGS="-shared"

if [ "{target_os}" = "qnx" ]; then
    CFLAGS="$CFLAGS -D_QNX_SOURCE"
    LDFLAGS="$LDFLAGS -lrt"
fi

# Create build directory
mkdir -p build
cd build

# Configure main project
cmake .. \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_C_COMPILER="$CC" \\
    -DCMAKE_CXX_COMPILER="$CXX" \\
    -DCMAKE_C_FLAGS="$CFLAGS" \\
    -DCMAKE_CXX_FLAGS="$CXXFLAGS" \\
    -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \\
    -DTARGET_PLATFORM="{platform}" \\
    -DTARGET_OS="{target_os}" \\
    -DBUILD_ALL_FEATURES=ON

# Build all
make -j$(nproc)

# Install
make install

# Create package
make package

echo "AUTOMOS AI build completed successfully"
"""
        
        script_path = build_path / "build.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        script_path.chmod(0o755)
        return script_path
    
    def create_deployment_package(self, platform, target_os, build_path):
        """Create deployment package for flashing"""
        
        print(f"📦 Creating deployment package for {platform} ({target_os})")
        
        package_name = f"automos_ai_{platform}_{target_os}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        package_path = build_path / package_name
        
        # Create package directory
        package_path.mkdir(exist_ok=True)
        
        # Copy built artifacts
        for feature_path in build_path.glob("*/"):
            if feature_path.is_dir() and feature_path.name not in [package_name]:
                feature_package_path = package_path / "features" / feature_path.name
                feature_package_path.mkdir(parents=True, exist_ok=True)
                
                # Copy built libraries
                for lib_file in feature_path.rglob("*.so"):
                    shutil.copy2(lib_file, feature_package_path)
                
                # Copy built executables
                for exe_file in feature_path.rglob("*.bin"):
                    shutil.copy2(exe_file, feature_package_path)
        
        # Create deployment scripts
        self.create_deployment_scripts(platform, target_os, package_path)
        
        # Create configuration files
        self.create_configuration_files(platform, target_os, package_path)
        
        # Create flashing instructions
        self.create_flashing_instructions(platform, target_os, package_path)
        
        # Create package archive
        archive_path = build_path / f"{package_name}.tar.gz"
        shutil.make_archive(str(archive_path.with_suffix('')), 'gztar', str(package_path))
        
        return {
            'package_name': package_name,
            'package_path': str(archive_path),
            'package_size': archive_path.stat().st_size,
            'status': 'success'
        }
    
    def create_deployment_scripts(self, platform, target_os, package_path):
        """Create deployment and flashing scripts"""
        
        scripts_dir = package_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Flash script
        flash_script = f"""#!/bin/bash
# AUTOMOS AI Flash Script
# Platform: {platform}
# Target OS: {target_os}

set -e

echo "Flashing AUTOMOS AI to {platform}"

# Check platform
PLATFORM="{platform}"
TARGET_OS="{target_os}"

# Flash based on platform method
case "$PLATFORM" in
    "jetson_nano")
        echo "Flashing via SD card..."
        ./scripts/flash_sd_card.sh
        ;;
    "jetson_xavier"|"jetson_orin")
        echo "Flashing via USB..."
        ./scripts/flash_usb.sh
        ;;
    "drive_agx")
        echo "Flashing via network..."
        ./scripts/flash_network.sh
        ;;
    "industrial_pc")
        echo "Flashing via USB boot..."
        ./scripts/flash_usb_boot.sh
        ;;
    *)
        echo "Unknown platform: $PLATFORM"
        exit 1
        ;;
esac

echo "Flash completed successfully"
"""
        
        flash_path = scripts_dir / "flash.sh"
        with open(flash_path, 'w') as f:
            f.write(flash_script)
        flash_path.chmod(0o755)
        
        # Platform-specific flash scripts
        self.create_platform_flash_scripts(platform, target_os, scripts_dir)
    
    def create_platform_flash_scripts(self, platform, target_os, scripts_dir):
        """Create platform-specific flash scripts"""
        
        if platform in ['jetson_nano']:
            # SD card flash script
            sd_script = """#!/bin/bash
# SD Card Flash Script for Jetson Nano

echo "Flashing AUTOMOS AI to SD card..."

# Check for SD card
SD_CARD="/dev/mmcblk0"
if [ ! -b "$SD_CARD" ]; then
    echo "SD card not found at $SD_CARD"
    echo "Available block devices:"
    lsblk
    exit 1
fi

# Unmount if mounted
sudo umount ${{SD_CARD}}* 2>/dev/null || true

# Flash image
echo "Writing image to SD card..."
sudo dd if=automos_ai_image.img of=$SD_CARD bs=4M conv=fsync status=progress

echo "SD card flash completed"
"""
            
            sd_path = scripts_dir / "flash_sd_card.sh"
            with open(sd_path, 'w') as f:
                f.write(sd_script)
            sd_path.chmod(0o755)
        
        elif platform in ['jetson_xavier', 'jetson_orin']:
            # USB flash script
            usb_script = f"""#!/bin/bash
# USB Flash Script for Jetson Xavier/Orin

echo "Flashing AUTOMOS AI via USB..."

# Put device in recovery mode
echo "Please put device in recovery mode:"
echo "1. Power off the device"
echo "2. Connect USB cable"
echo "3. Press recovery button"
echo "4. Power on device"
echo ""
read -p "Press Enter when device is in recovery mode..."

# Flash using NVIDIA SDK
cd /opt/nvidia/sdkmanager/
sudo ./sdkmanager --cli install \
    --device jetson-{platform} \
    --flash os \
    --image automos_ai_image.img

echo "USB flash completed"
"""
            
            usb_path = scripts_dir / "flash_usb.sh"
            with open(usb_path, 'w') as f:
                f.write(usb_script)
            usb_path.chmod(0o755)
    
    def create_configuration_files(self, platform, target_os, package_path):
        """Create configuration files"""
        
        config_dir = package_path / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Platform configuration
        platform_config = self.platforms[platform]
        
        config_data = {
            'platform': platform,
            'target_os': target_os,
            'architecture': platform_config['arch'],
            'features': platform_config['features'],
            'memory_limit': platform_config['memory_limit'],
            'flash_method': platform_config['flash_method'],
            'build_time': datetime.now().isoformat()
        }
        
        with open(config_dir / "platform_config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # AUTOMOS AI configuration
        automos_config = {
            'system': {
                'name': 'AUTOMOS AI',
                'version': '1.0.0',
                'platform': platform,
                'target_os': target_os
            },
            'perception': {
                'camera_count': 6,
                'radar_sensors': 4,
                'lidar_sensors': 2,
                'processing_fps': 30
            },
            'safety': {
                'response_time_ms': 10,
                'emergency_override': True,
                'collision_detection': True
            },
            'ai_reasoning': {
                'llm_model': 'dualad_opendrivevla_fusion',
                'planning_algorithm': 'lattice_idm_vla'
            },
            'edge_optimization': {
                'quantization': 'dynamic_int8',
                'memory_optimization': True,
                'performance_mode': 'balanced'
            }
        }
        
        with open(config_dir / "automos_config.json", 'w') as f:
            json.dump(automos_config, f, indent=2)
    
    def create_flashing_instructions(self, platform, target_os, package_path):
        """Create flashing instructions"""
        
        instructions = f"""# AUTOMOS AI Flashing Instructions

## Platform: {platform}
## Target OS: {target_os}
## Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Prerequisites
1. {platform} device
2. USB cable / SD card / Network connection
3. Host machine with AUTOMOS AI build tools
4. Sufficient storage for flashing

### Flashing Steps

#### 1. Prepare Device
- Power off the {platform} device
- Connect required cables
- Put device in flashing mode

#### 2. Run Flash Script
```bash
cd {package_path.name}
sudo ./scripts/flash.sh
```

#### 3. Verify Installation
- Power on device
- Check AUTOMOS AI service status
- Verify all components are running

### Troubleshooting

#### Flash Fails
- Check cable connections
- Verify device is in correct mode
- Ensure sufficient storage space

#### Boot Issues
- Check flash integrity
- Verify boot configuration
- Check hardware compatibility

#### Service Issues
- Check system logs
- Verify configuration files
- Restart AUTOMOS AI services

### Support
For support, contact: support@automos-ai.com
"""
        
        instructions_path = package_path / "FLASHING_INSTRUCTIONS.md"
        with open(instructions_path, 'w') as f:
            f.write(instructions)
    
    def clean_build(self, platform):
        """Clean build directory for platform"""
        
        build_path = self.build_dir / platform
        if build_path.exists():
            shutil.rmtree(build_path)
            print(f"🧹 Cleaned build directory: {build_path}")
    
    def list_platforms(self):
        """List supported platforms"""
        
        print("🎯 Supported Platforms:")
        for platform, config in self.platforms.items():
            print(f"  {platform}:")
            print(f"    Architecture: {config['arch']}")
            print(f"    QNX Support: {config['qnx_support']}")
            print(f"    Features: {', '.join(config['features'])}")
            print(f"    Flash Method: {config['flash_method']}")
            print()

def main():
    """Main build system interface"""
    
    parser = argparse.ArgumentParser(description='AUTOMOS AI Build System')
    parser.add_argument('--platform', 
                       choices=['jetson_nano', 'jetson_xavier', 'jetson_orin', 'drive_agx', 'industrial_pc'],
                       help='Target platform')
    parser.add_argument('--target-os', default='linux', choices=['linux', 'qnx'],
                       help='Target operating system')
    parser.add_argument('--features', nargs='+', 
                       help='Specific features to build')
    parser.add_argument('--clean', action='store_true',
                       help='Clean build before building')
    parser.add_argument('--list-platforms', action='store_true',
                       help='List supported platforms')
    
    args = parser.parse_args()
    
    build_system = AUTOMOSBuildSystem()
    
    if args.list_platforms:
        build_system.list_platforms()
        return
    
    if not args.platform:
        parser.error("--platform is required unless --list-platforms is used")
    
    try:
        result = build_system.build(
            platform=args.platform,
            features=args.features,
            target_os=args.target_os,
            clean=args.clean
        )
        
        print(f"\n🎉 Build completed!")
        print(f"📊 Status: {result['status']}")
        print(f"📦 Package: {result['package_result']['package_path']}")
        
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
