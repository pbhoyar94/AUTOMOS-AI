#!/usr/bin/env python3
"""
Unreal Engine Installation Guide for CARLA
Automated setup for Unreal Engine to build CARLA
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_epic_launcher():
    """Check if Epic Games Launcher is running"""
    try:
        # Check if Epic Games Launcher process is running
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq EpicGamesLauncher.exe'], 
                              capture_output=True, text=True)
        return 'EpicGamesLauncher.exe' in result.stdout
    except Exception as e:
        logger.error(f"Error checking Epic Games Launcher: {e}")
        return False

def launch_epic_launcher():
    """Launch Epic Games Launcher"""
    launcher_path = Path("C:/Program Files/Epic Games/Launcher/Portal/Binaries/Win64/EpicGamesLauncher.exe")
    
    if not launcher_path.exists():
        logger.error("Epic Games Launcher not found")
        return False
    
    try:
        logger.info("Launching Epic Games Launcher...")
        subprocess.Popen([str(launcher_path)])
        
        # Wait for launcher to start
        for i in range(30):  # Wait up to 30 seconds
            if check_epic_launcher():
                logger.info("✅ Epic Games Launcher started successfully")
                return True
            time.sleep(1)
        
        logger.error("❌ Epic Games Launcher failed to start")
        return False
        
    except Exception as e:
        logger.error(f"Failed to launch Epic Games Launcher: {e}")
        return False

def install_unreal_engine_guide():
    """Provide step-by-step guide for Unreal Engine installation"""
    logger.info("="*60)
    logger.info("UNREAL ENGINE INSTALLATION GUIDE")
    logger.info("="*60)
    
    logger.info("\n📋 STEPS TO INSTALL UNREAL ENGINE:")
    logger.info("1. Epic Games Launcher should now be open")
    logger.info("2. Click on 'Unreal Engine' tab in the left sidebar")
    logger.info("3. Click 'Install' on Unreal Engine 4.27 (or latest 4.x version)")
    logger.info("4. Choose installation location:")
    logger.info("   - Recommended: C:/Program Files/Epic Games/UE_4.27")
    logger.info("   - Or any location with at least 20GB free space")
    logger.info("5. Click 'Install' and wait for download to complete")
    logger.info("6. Once installed, return to this script")
    
    logger.info("\n⚠️ IMPORTANT NOTES:")
    logger.info("- Unreal Engine 4.24+ is required for CARLA")
    logger.info("- UE 4.27 is recommended for best compatibility")
    logger.info("- Installation may take 30-60 minutes")
    logger.info("- You need at least 20GB free disk space")
    
    logger.info("\n🔄 AFTER INSTALLATION:")
    logger.info("- Run this script again to verify installation")
    logger.info("- Then build CARLA with: cd ../carla && ./Update.bat")
    
    return True

def check_unreal_engine_installation():
    """Check if Unreal Engine is installed"""
    logger.info("Checking Unreal Engine installation...")
    
    # Common installation paths
    ue_paths = [
        Path("C:/Program Files/Epic Games/UE_4.27"),
        Path("C:/Program Files/Epic Games/UE_4.26"),
        Path("C:/Program Files/Epic Games/UE_4.25"),
        Path("C:/Program Files/Epic Games/UE_4.24"),
        Path("C:/Program Files (x86)/Epic Games/UE_4.27"),
        Path("C:/Program Files (x86)/Epic Games/UE_4.26"),
        Path("C:/Program Files (x86)/Epic Games/UE_4.25"),
        Path("C:/Program Files (x86)/Epic Games/UE_4.24"),
        Path.home() / "UnrealEngine" / "UE_4.27",
        Path.home() / "UnrealEngine" / "UE_4.26",
        Path.home() / "UnrealEngine" / "UE_4.25",
        Path.home() / "UnrealEngine" / "UE_4.24"
    ]
    
    for ue_path in ue_paths:
        if ue_path.exists():
            # Check for UE4 executable
            ue4_exe = ue_path / "Engine" / "Binaries" / "Win64" / "UE4Editor.exe"
            if ue4_exe.exists():
                logger.info(f"✅ Unreal Engine found: {ue_path}")
                logger.info(f"✅ UE4Editor.exe found: {ue4_exe}")
                return ue_path, ue4_exe
    
    logger.error("❌ Unreal Engine not found")
    return None, None

def build_carla_with_ue(ue_path):
    """Build CARLA using Unreal Engine"""
    carla_path = Path(__file__).parent.parent / "carla"
    
    if not carla_path.exists():
        logger.error("❌ CARLA directory not found")
        return False
    
    try:
        logger.info("Building CARLA with Unreal Engine...")
        logger.info(f"CARLA path: {carla_path}")
        logger.info(f"Unreal Engine path: {ue_path}")
        
        # Set environment variables
        env = os.environ.copy()
        env['UNREAL_ENGINE_PATH'] = str(ue_path)
        
        # Run CARLA update script
        update_script = carla_path / "Update.bat"
        
        if update_script.exists():
            logger.info("Running CARLA Update.bat...")
            result = subprocess.run([str(update_script)], 
                                  cwd=str(carla_path), 
                                  env=env,
                                  capture_output=True, 
                                  text=True)
            
            if result.returncode == 0:
                logger.info("✅ CARLA build completed successfully")
                return True
            else:
                logger.error(f"❌ CARLA build failed: {result.stderr}")
                return False
        else:
            logger.error("❌ Update.bat not found in CARLA directory")
            return False
            
    except Exception as e:
        logger.error(f"Failed to build CARLA: {e}")
        return False

def main():
    """Main installation process"""
    logger.info("Unreal Engine Installation for CARLA")
    logger.info("="*60)
    
    # Check current status
    ue_path, ue4_exe = check_unreal_engine_installation()
    
    if ue_path and ue4_exe:
        logger.info("✅ Unreal Engine is already installed!")
        logger.info("Building CARLA...")
        
        if build_carla_with_ue(ue_path):
            logger.info("🎉 CARLA build completed successfully!")
            logger.info("You can now start CARLA server with:")
            logger.info("python start_carla_server.py")
            return 0
        else:
            logger.error("❌ CARLA build failed")
            return 1
    
    else:
        logger.info("❌ Unreal Engine not found")
        logger.info("Starting installation process...")
        
        # Launch Epic Games Launcher
        if not check_epic_launcher():
            if not launch_epic_launcher():
                logger.error("Failed to launch Epic Games Launcher")
                return 1
        
        # Show installation guide
        install_unreal_engine_guide()
        
        logger.info("\n" + "="*60)
        logger.info("NEXT STEPS:")
        logger.info("1. Follow the installation guide above")
        logger.info("2. Install Unreal Engine 4.27")
        logger.info("3. Run this script again to build CARLA")
        logger.info("4. Start CARLA server and test AUTOMOS AI")
        logger.info("="*60)
        
        return 0

if __name__ == "__main__":
    sys.exit(main())
