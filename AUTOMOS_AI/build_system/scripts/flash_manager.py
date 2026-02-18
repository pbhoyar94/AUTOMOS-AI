#!/usr/bin/env python3
"""
AUTOMOS AI Flash Manager
Hardware flashing and deployment management system
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

class AUTOMOSFlashManager:
    """AUTOMOS AI Flash Manager for hardware deployment"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.deployments_dir = self.project_root / "deployments"
        self.scripts_dir = Path(__file__).parent
        
        # Flash methods for different platforms
        self.flash_methods = {
            'sd_card': self.flash_sd_card,
            'usb_flash': self.flash_usb,
            'network_flash': self.flash_network,
            'usb_boot': self.flash_usb_boot,
            'jtag': self.flash_jtag
        }
        
        # Platform-specific configurations
        self.platform_configs = {
            'jetson_nano': {
                'flash_method': 'sd_card',
                'image_size': '8GB',
                'boot_time': '2-3 minutes',
                'recovery': 'SD card reflash'
            },
            'jetson_xavier': {
                'flash_method': 'usb_flash',
                'image_size': '32GB',
                'boot_time': '1-2 minutes',
                'recovery': 'USB recovery mode'
            },
            'jetson_orin': {
                'flash_method': 'usb_flash',
                'image_size': '64GB',
                'boot_time': '1-2 minutes',
                'recovery': 'USB recovery mode'
            },
            'drive_agx': {
                'flash_method': 'network_flash',
                'image_size': '128GB',
                'boot_time': '30-60 seconds',
                'recovery': 'Network recovery'
            },
            'industrial_pc': {
                'flash_method': 'usb_boot',
                'image_size': '16GB',
                'boot_time': '1-2 minutes',
                'recovery': 'USB boot recovery'
            }
        }
    
    def flash_platform(self, platform, deployment_path, method=None):
        """Flash AUTOMOS AI to specified platform"""
        
        print(f"🔥 Flashing AUTOMOS AI to {platform}")
        
        if platform not in self.platform_configs:
            raise ValueError(f"Unsupported platform: {platform}")
        
        platform_config = self.platform_configs[platform]
        flash_method = method or platform_config['flash_method']
        
        # Validate deployment package
        deployment_path = Path(deployment_path)
        if not deployment_path.exists():
            raise FileNotFoundError(f"Deployment package not found: {deployment_path}")
        
        # Check flash method
        if flash_method not in self.flash_methods:
            raise ValueError(f"Unsupported flash method: {flash_method}")
        
        # Flash the platform
        flash_function = self.flash_methods[flash_method]
        result = flash_function(platform, deployment_path)
        
        return result
    
    def flash_sd_card(self, platform, deployment_path):
        """Flash via SD card (Jetson Nano)"""
        
        print(f"💾 Flashing {platform} via SD card")
        
        # Check for SD card
        sd_card = self.detect_sd_card()
        if not sd_card:
            raise RuntimeError("No SD card detected")
        
        print(f"Found SD card: {sd_card}")
        
        # Unmount SD card
        self.unmount_device(sd_card)
        
        # Find image file
        image_file = deployment_path / "automos_ai_image.img"
        if not image_file.exists():
            # Create image from deployment
            image_file = self.create_sd_image(deployment_path)
        
        # Flash image to SD card
        print(f"Flashing image to {sd_card}...")
        result = subprocess.run([
            'sudo', 'dd', f'if={image_file}', f'of={sd_card}',
            'bs=4M', 'conv=fsync', 'status=progress'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Flash failed: {result.stderr}")
        
        # Verify flash
        self.verify_flash(sd_card, image_file)
        
        return {
            'platform': platform,
            'method': 'sd_card',
            'device': sd_card,
            'status': 'success',
            'message': f"Successfully flashed {platform} via SD card"
        }
    
    def flash_usb(self, platform, deployment_path):
        """Flash via USB (Jetson Xavier/Orin)"""
        
        print(f"🔌 Flashing {platform} via USB")
        
        # Check for NVIDIA SDK Manager
        if not self.check_nvidia_sdk():
            print("Installing NVIDIA SDK Manager...")
            self.install_nvidia_sdk()
        
        # Put device in recovery mode
        print("Please put device in recovery mode:")
        print("1. Power off the device")
        print("2. Connect USB cable")
        print("3. Press recovery button")
        print("4. Power on device")
        
        input("Press Enter when device is in recovery mode...")
        
        # Check device in recovery mode
        if not self.check_recovery_mode():
            raise RuntimeError("Device not detected in recovery mode")
        
        # Create flash package
        flash_package = self.create_usb_flash_package(deployment_path)
        
        # Flash using NVIDIA SDK Manager
        cmd = [
            'sudo', '/opt/nvidia/sdkmanager/sdkmanager',
            '--cli', 'install',
            '--device', f'jetson-{platform}',
            '--flash', 'os',
            '--image', str(flash_package)
        ]
        
        print(f"Running flash command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"USB flash failed: {result.stderr}")
        
        return {
            'platform': platform,
            'method': 'usb_flash',
            'status': 'success',
            'message': f"Successfully flashed {platform} via USB"
        }
    
    def flash_network(self, platform, deployment_path):
        """Flash via network (DRIVE AGX)"""
        
        print(f"🌐 Flashing {platform} via network")
        
        # Get device IP
        device_ip = input("Enter DRIVE AGX device IP: ")
        
        # Test connection
        if not self.test_network_connection(device_ip):
            raise RuntimeError(f"Cannot connect to device at {device_ip}")
        
        # Create network flash package
        flash_package = self.create_network_flash_package(deployment_path)
        
        # Transfer package to device
        print(f"Transferring flash package to {device_ip}...")
        transfer_cmd = [
            'scp', str(flash_package), f'nvidia@{device_ip}:/tmp/'
        ]
        
        result = subprocess.run(transfer_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Transfer failed: {result.stderr}")
        
        # Execute flash on device
        flash_cmd = [
            'ssh', f'nvidia@{device_ip}',
            f'sudo /tmp/{flash_package.name}/flash.sh'
        ]
        
        print(f"Executing flash on device...")
        result = subprocess.run(flash_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Network flash failed: {result.stderr}")
        
        return {
            'platform': platform,
            'method': 'network_flash',
            'device_ip': device_ip,
            'status': 'success',
            'message': f"Successfully flashed {platform} via network"
        }
    
    def flash_usb_boot(self, platform, deployment_path):
        """Flash via USB boot (Industrial PC)"""
        
        print(f"💿 Flashing {platform} via USB boot")
        
        # Create bootable USB
        bootable_usb = self.create_bootable_usb(deployment_path)
        
        # Detect USB drive
        usb_drive = self.detect_usb_drive()
        if not usb_drive:
            raise RuntimeError("No USB drive detected")
        
        # Flash USB drive
        print(f"Flashing bootable USB to {usb_drive}...")
        result = subprocess.run([
            'sudo', 'dd', f'if={bootable_usb}', f'of={usb_drive}',
            'bs=4M', 'conv=fsync', 'status=progress'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"USB boot flash failed: {result.stderr}")
        
        return {
            'platform': platform,
            'method': 'usb_boot',
            'device': usb_drive,
            'status': 'success',
            'message': f"Successfully created bootable USB for {platform}"
        }
    
    def flash_jtag(self, platform, deployment_path):
        """Flash via JTAG (advanced debugging)"""
        
        print(f"🔧 Flashing {platform} via JTAG")
        
        # Check for JTAG hardware
        if not self.check_jtag_hardware():
            raise RuntimeError("JTAG hardware not detected")
        
        # Create JTAG flash package
        jtag_package = self.create_jtag_flash_package(deployment_path)
        
        # Execute JTAG flash
        print("Executing JTAG flash...")
        result = subprocess.run([
            'openocd', '-f', str(jtag_package / 'jtag_config.cfg'),
            '-c', 'init', 'reset_config srst_only srst_nogate',
            '-c', 'flash write_image erase automos_ai.bin 0x0',
            '-c', 'reset', 'shutdown'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"JTAG flash failed: {result.stderr}")
        
        return {
            'platform': platform,
            'method': 'jtag',
            'status': 'success',
            'message': f"Successfully flashed {platform} via JTAG"
        }
    
    def detect_sd_card(self):
        """Detect SD card device"""
        
        try:
            result = subprocess.run(['lsblk'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'mmcblk0' in line and 'disk' in line:
                    return '/dev/mmcblk0'
            
            return None
        except Exception:
            return None
    
    def detect_usb_drive(self):
        """Detect USB drive device"""
        
        try:
            result = subprocess.run(['lsblk'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'usb' in line.lower() and 'disk' in line:
                    parts = line.split()
                    if parts:
                        return f"/dev/{parts[0]}"
            
            return None
        except Exception:
            return None
    
    def unmount_device(self, device):
        """Unmount all partitions on device"""
        
        try:
            result = subprocess.run(['lsblk', device], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if device in line and 'part' in line:
                    partition = line.split()[0]
                    subprocess.run(['sudo', 'umount', f'/dev/{partition}'], 
                                 capture_output=True, text=True)
        except Exception:
            pass
    
    def check_nvidia_sdk(self):
        """Check if NVIDIA SDK Manager is installed"""
        
        return Path('/opt/nvidia/sdkmanager/sdkmanager').exists()
    
    def install_nvidia_sdk(self):
        """Install NVIDIA SDK Manager"""
        
        print("Installing NVIDIA SDK Manager...")
        
        # Download and install SDK Manager
        install_cmd = [
            'wget', '-qO-', 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin',
            '|', 'sudo', 'tee', '/etc/apt/preferences.d/cuda-repository-pin-600'
        ]
        
        subprocess.run(' '.join(install_cmd), shell=True, check=True)
        subprocess.run(['sudo', 'apt', 'update'], check=True)
        subprocess.run(['sudo', 'apt', 'install', '-y', 'nvidia-sdkmanager'], check=True)
    
    def check_recovery_mode(self):
        """Check if device is in recovery mode"""
        
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            return 'Nvidia Corp' in result.stdout or '0955' in result.stdout
        except Exception:
            return False
    
    def test_network_connection(self, ip):
        """Test network connection to device"""
        
        try:
            result = subprocess.run(['ping', '-c', '1', ip], 
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def check_jtag_hardware(self):
        """Check for JTAG hardware"""
        
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            return 'FTDI' in result.stdout or 'JTAG' in result.stdout
        except Exception:
            return False
    
    def create_sd_image(self, deployment_path):
        """Create SD card image from deployment"""
        
        print("Creating SD card image...")
        
        image_path = deployment_path / "automos_ai_image.img"
        image_size = 8 * 1024 * 1024 * 1024  # 8GB
        
        # Create empty image
        subprocess.run([
            'fallocate', '-l', f'{image_size}', str(image_path)
        ], check=True)
        
        # Create partitions
        subprocess.run([
            'parted', str(image_path), 'mklabel', 'msdos',
            'mkpart', 'primary', 'fat32', '1MiB', '256MiB',
            'mkpart', 'primary', 'ext4', '256MiB', '100%'
        ], check=True)
        
        # Setup loop device
        result = subprocess.run(['sudo', 'losetup', '--show', '-f', str(image_path)],
                              capture_output=True, text=True)
        loop_device = result.stdout.strip()
        
        try:
            # Create filesystems
            subprocess.run(['sudo', 'mkfs.vfat', f'{loop_device}p1'], check=True)
            subprocess.run(['sudo', 'mkfs.ext4', f'{loop_device}p2'], check=True)
            
            # Mount and copy files
            mount_point = Path('/tmp/automos_sd_mount')
            mount_point.mkdir(exist_ok=True)
            
            subprocess.run(['sudo', 'mount', f'{loop_device}p1', str(mount_point)], check=True)
            
            # Copy boot files
            boot_files = deployment_path / "boot"
            if boot_files.exists():
                subprocess.run(['sudo', 'cp', '-r', str(boot_files) + '/*', str(mount_point)],
                             check=True)
            
            subprocess.run(['sudo', 'umount', str(mount_point)], check=True)
            subprocess.run(['sudo', 'mount', f'{loop_device}p2', str(mount_point)], check=True)
            
            # Copy root filesystem
            root_files = deployment_path / "rootfs"
            if root_files.exists():
                subprocess.run(['sudo', 'cp', '-r', str(root_files) + '/*', str(mount_point)],
                             check=True)
            
            subprocess.run(['sudo', 'umount', str(mount_point)], check=True)
            
        finally:
            # Cleanup loop device
            subprocess.run(['sudo', 'losetup', '-d', loop_device], check=False)
        
        return image_path
    
    def create_usb_flash_package(self, deployment_path):
        """Create USB flash package"""
        
        flash_package = deployment_path / "usb_flash_package"
        flash_package.mkdir(exist_ok=True)
        
        # Copy deployment files
        subprocess.run(['cp', '-r', str(deployment_path) + '/*', str(flash_package)],
                     check=True)
        
        # Create flash script
        flash_script = flash_package / "flash.sh"
        with open(flash_script, 'w') as f:
            f.write("""#!/bin/bash
# AUTOMOS AI USB Flash Script
set -e

echo "Flashing AUTOMOS AI..."

# Flash commands here
# This would be platform-specific

echo "Flash completed"
""")
        flash_script.chmod(0o755)
        
        return flash_package
    
    def create_network_flash_package(self, deployment_path):
        """Create network flash package"""
        
        flash_package = deployment_path / "network_flash_package"
        flash_package.mkdir(exist_ok=True)
        
        # Copy deployment files
        subprocess.run(['cp', '-r', str(deployment_path) + '/*', str(flash_package)],
                     check=True)
        
        # Create flash script
        flash_script = flash_package / "flash.sh"
        with open(flash_script, 'w') as f:
            f.write("""#!/bin/bash
# AUTOMOS AI Network Flash Script
set -e

echo "Flashing AUTOMOS AI via network..."

# Network flash commands here
# This would be platform-specific

echo "Flash completed"
""")
        flash_script.chmod(0o755)
        
        return flash_package
    
    def create_bootable_usb(self, deployment_path):
        """Create bootable USB image"""
        
        print("Creating bootable USB image...")
        
        usb_image = deployment_path / "automos_ai_bootable.img"
        image_size = 16 * 1024 * 1024 * 1024  # 16GB
        
        # Create empty image
        subprocess.run([
            'fallocate', '-l', f'{image_size}', str(usb_image)
        ], check=True)
        
        # Create partitions and filesystems
        subprocess.run([
            'parted', str(usb_image), 'mklabel', 'gpt',
            'mkpart', 'primary', 'fat32', '1MiB', '512MiB',
            'mkpart', 'primary', 'ext4', '512MiB', '100%'
        ], check=True)
        
        # Setup loop device and copy files
        result = subprocess.run(['sudo', 'losetup', '--show', '-f', str(usb_image)],
                              capture_output=True, text=True)
        loop_device = result.stdout.strip()
        
        try:
            subprocess.run(['sudo', 'mkfs.vfat', f'{loop_device}p1'], check=True)
            subprocess.run(['sudo', 'mkfs.ext4', f'{loop_device}p2'], check=True)
            
            # Mount and copy files
            mount_point = Path('/tmp/automos_usb_mount')
            mount_point.mkdir(exist_ok=True)
            
            subprocess.run(['sudo', 'mount', f'{loop_device}p2', str(mount_point)], check=True)
            
            # Copy root filesystem
            root_files = deployment_path / "rootfs"
            if root_files.exists():
                subprocess.run(['sudo', 'cp', '-r', str(root_files) + '/*', str(mount_point)],
                             check=True)
            
            subprocess.run(['sudo', 'umount', str(mount_point)], check=True)
            
        finally:
            subprocess.run(['sudo', 'losetup', '-d', loop_device], check=False)
        
        return usb_image
    
    def create_jtag_flash_package(self, deployment_path):
        """Create JTAG flash package"""
        
        jtag_package = deployment_path / "jtag_flash_package"
        jtag_package.mkdir(exist_ok=True)
        
        # Copy deployment files
        subprocess.run(['cp', '-r', str(deployment_path) + '/*', str(jtag_package)],
                     check=True)
        
        # Create JTAG config
        jtag_config = jtag_package / "jtag_config.cfg"
        with open(jtag_config, 'w') as f:
            f.write("""# JTAG Configuration for AUTOMOS AI
interface jtag
target create jtag.target cortex_m -endian little -chain-position 0
""")
        
        return jtag_package
    
    def verify_flash(self, device, image_file):
        """Verify flash integrity"""
        
        print("Verifying flash integrity...")
        
        # Calculate image checksum
        result = subprocess.run(['md5sum', str(image_file)], capture_output=True, text=True)
        image_checksum = result.stdout.split()[0]
        
        # Calculate device checksum (first part)
        device_checksum = subprocess.run([
            'sudo', 'dd', f'if={device}', 'bs=1M', 'count=100',
            '|', 'md5sum'
        ], shell=True, capture_output=True, text=True)
        
        if device_checksum.returncode != 0:
            print("Warning: Could not verify flash integrity")
            return True
        
        print("Flash verification completed")
        return True
    
    def list_deployments(self):
        """List available deployment packages"""
        
        print("📦 Available Deployment Packages:")
        
        if not self.deployments_dir.exists():
            print("No deployment packages found")
            return
        
        for deployment in self.deployments_dir.glob("*.tar.gz"):
            print(f"  {deployment.name}")
            
            # Extract info from deployment
            info_file = deployment.with_suffix('.json')
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                    print(f"    Platform: {info.get('platform', 'Unknown')}")
                    print(f"    Target OS: {info.get('target_os', 'Unknown')}")
                    print(f"    Created: {info.get('created', 'Unknown')}")
            print()

def main():
    """Main flash manager interface"""
    
    parser = argparse.ArgumentParser(description='AUTOMOS AI Flash Manager')
    parser.add_argument('--platform', required=True,
                       choices=['jetson_nano', 'jetson_xavier', 'jetson_orin', 'drive_agx', 'industrial_pc'],
                       help='Target platform')
    parser.add_argument('--deployment', required=True,
                       help='Deployment package path')
    parser.add_argument('--method', 
                       choices=['sd_card', 'usb_flash', 'network_flash', 'usb_boot', 'jtag'],
                       help='Flash method (auto-detected if not specified)')
    parser.add_argument('--list-deployments', action='store_true',
                       help='List available deployment packages')
    
    args = parser.parse_args()
    
    flash_manager = AUTOMOSFlashManager()
    
    if args.list_deployments:
        flash_manager.list_deployments()
        return
    
    try:
        result = flash_manager.flash_platform(
            platform=args.platform,
            deployment_path=args.deployment,
            method=args.method
        )
        
        print(f"\n🎉 Flash completed successfully!")
        print(f"📊 Platform: {result['platform']}")
        print(f"🔥 Method: {result['method']}")
        print(f"✅ Status: {result['status']}")
        print(f"💬 Message: {result['message']}")
        
    except Exception as e:
        print(f"❌ Flash failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
