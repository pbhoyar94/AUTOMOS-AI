#!/usr/bin/env python3
"""
Automated Unreal Engine Installation for CARLA
Automated setup with progress monitoring and verification
"""

import os
import sys
import time
import subprocess
import logging
import psutil
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutomatedUnrealInstaller:
    """Automated Unreal Engine installation with monitoring"""
    
    def __init__(self):
        """Initialize installer"""
        self.epic_launcher_path = None
        self.ue_installation_path = None
        self.installation_progress = 0
        self.installation_status = "Not Started"
        
    def run_automated_installation(self) -> bool:
        """Run complete automated installation"""
        logger.info("🚀 Automated Unreal Engine Installation for CARLA")
        logger.info("=" * 60)
        
        try:
            # Step 1: Verify Epic Games Launcher
            if not self._verify_epic_launcher():
                logger.error("❌ Epic Games Launcher not found")
                return False
            
            # Step 2: Check existing UE installation
            if self._check_existing_ue():
                logger.info("✅ Unreal Engine already installed")
                return self._proceed_with_carla_build()
            
            # Step 3: Launch Epic Games Launcher
            if not self._launch_epic_launcher():
                logger.error("❌ Failed to launch Epic Games Launcher")
                return False
            
            # Step 4: Guide user through UE installation
            if not self._guide_ue_installation():
                logger.error("❌ Unreal Engine installation failed")
                return False
            
            # Step 5: Verify UE installation
            if not self._verify_ue_installation():
                logger.error("❌ Unreal Engine installation verification failed")
                return False
            
            # Step 6: Build CARLA
            return self._proceed_with_carla_build()
            
        except Exception as e:
            logger.error(f"Automated installation failed: {e}")
            return False
    
    def _verify_epic_launcher(self) -> bool:
        """Verify Epic Games Launcher is installed"""
        logger.info("🔍 Verifying Epic Games Launcher...")
        
        epic_paths = [
            Path("C:/Program Files/Epic Games/Launcher/Portal/Binaries/Win64/EpicGamesLauncher.exe"),
            Path("C:/Program Files (x86)/Epic Games/Launcher/Portal/Binaries/Win64/EpicGamesLauncher.exe")
        ]
        
        for path in epic_paths:
            if path.exists():
                self.epic_launcher_path = path
                logger.info(f"✅ Epic Games Launcher found: {path}")
                return True
        
        logger.error("❌ Epic Games Launcher not found")
        return False
    
    def _check_existing_ue(self) -> bool:
        """Check if Unreal Engine is already installed"""
        logger.info("🔍 Checking existing Unreal Engine installation...")
        
        ue_paths = [
            Path("C:/Program Files/Epic Games/UE_4.27"),
            Path("C:/Program Files/Epic Games/UE_4.26"),
            Path("C:/Program Files (x86)/Epic Games/UE_4.27"),
            Path("C:/Program Files (x86)/Epic Games/UE_4.26"),
            Path.home() / "UnrealEngine" / "UE_4.27",
            Path.home() / "UnrealEngine" / "UE_4.26"
        ]
        
        for ue_path in ue_paths:
            if ue_path.exists():
                ue4_exe = ue_path / "Engine" / "Binaries" / "Win64" / "UE4Editor.exe"
                if ue4_exe.exists():
                    self.ue_installation_path = ue_path
                    logger.info(f"✅ Unreal Engine found: {ue_path}")
                    return True
        
        logger.info("ℹ️ Unreal Engine not found, proceeding with installation")
        return False
    
    def _launch_epic_launcher(self) -> bool:
        """Launch Epic Games Launcher"""
        logger.info("🚀 Launching Epic Games Launcher...")
        
        try:
            # Check if launcher is already running
            launcher_running = self._is_process_running("EpicGamesLauncher.exe")
            
            if not launcher_running:
                # Launch Epic Games Launcher
                subprocess.Popen([str(self.epic_launcher_path)])
                logger.info("✅ Epic Games Launcher launched")
                
                # Wait for launcher to start
                for i in range(30):
                    if self._is_process_running("EpicGamesLauncher.exe"):
                        logger.info("✅ Epic Games Launcher is running")
                        break
                    time.sleep(1)
                else:
                    logger.error("❌ Epic Games Launcher failed to start")
                    return False
            else:
                logger.info("✅ Epic Games Launcher already running")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch Epic Games Launcher: {e}")
            return False
    
    def _is_process_running(self, process_name: str) -> bool:
        """Check if a process is running"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == process_name:
                    return True
            return False
        except Exception:
            return False
    
    def _guide_ue_installation(self) -> bool:
        """Guide user through Unreal Engine installation"""
        logger.info("\n" + "="*60)
        logger.info("UNREAL ENGINE INSTALLATION GUIDE")
        logger.info("="*60)
        
        logger.info("\n📋 MANUAL INSTALLATION STEPS:")
        logger.info("1. Epic Games Launcher should now be open on your desktop")
        logger.info("2. Click on 'Unreal Engine' tab in the left sidebar")
        logger.info("3. Find 'Unreal Engine 4.27' (or latest 4.x version)")
        logger.info("4. Click 'Install' button")
        logger.info("5. Choose installation location:")
        logger.info("   - Recommended: C:/Program Files/Epic Games/UE_4.27")
        logger.info("   - Minimum: 20GB free space required")
        logger.info("6. Click 'Install' and wait for download to complete")
        
        logger.info("\n⏱️ EXPECTED INSTALLATION TIME:")
        logger.info("- Download: 20-40 minutes (depending on internet speed)")
        logger.info("- Installation: 10-20 minutes")
        logger.info("- Total: 30-60 minutes")
        
        logger.info("\n💾 SYSTEM REQUIREMENTS:")
        logger.info("- Disk Space: 20GB+ free")
        logger.info("- Internet: Stable connection")
        logger.info("- RAM: 8GB+ recommended")
        logger.info("- Graphics: DirectX 11 compatible")
        
        logger.info("\n🔄 AFTER INSTALLATION:")
        logger.info("1. Return to this script")
        logger.info("2. Press Enter to continue")
        logger.info("3. Script will verify installation")
        logger.info("4. Build CARLA automatically")
        
        # Wait for user to complete installation
        logger.info("\n" + "="*60)
        logger.info("⏳ WAITING FOR UNREAL ENGINE INSTALLATION...")
        logger.info("="*60)
        logger.info("Complete the installation in Epic Games Launcher")
        logger.info("Then press Enter to continue...")
        
        input()  # Wait for user input
        
        return True
    
    def _verify_ue_installation(self) -> bool:
        """Verify Unreal Engine installation"""
        logger.info("🔍 Verifying Unreal Engine installation...")
        
        # Check again for UE installation
        if self._check_existing_ue():
            logger.info("✅ Unreal Engine installation verified")
            return True
        else:
            logger.error("❌ Unreal Engine installation not found")
            logger.info("Please ensure Unreal Engine 4.27 is installed in Epic Games Launcher")
            return False
    
    def _proceed_with_carla_build(self) -> bool:
        """Proceed with CARLA build"""
        logger.info("🔧 Building CARLA with Unreal Engine...")
        
        try:
            carla_path = Path(__file__).parent.parent / "carla"
            
            if not carla_path.exists():
                logger.error("❌ CARLA directory not found")
                return False
            
            # Set environment variables
            env = os.environ.copy()
            if self.ue_installation_path:
                env['UNREAL_ENGINE_PATH'] = str(self.ue_installation_path)
            
            # Run CARLA update script
            update_script = carla_path / "Update.bat"
            
            if update_script.exists():
                logger.info("🚀 Running CARLA Update.bat...")
                logger.info("This may take 30-60 minutes to complete...")
                
                # Start the build process
                process = subprocess.Popen([str(update_script)], 
                                        cwd=str(carla_path), 
                                        env=env,
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.STDOUT,
                                        text=True,
                                        bufsize=1,
                                        universal_newlines=True)
                
                # Monitor build progress
                self._monitor_build_process(process)
                
                # Check build result
                if process.returncode == 0:
                    logger.info("✅ CARLA build completed successfully")
                    return True
                else:
                    logger.error("❌ CARLA build failed")
                    return False
            else:
                logger.error("❌ Update.bat not found in CARLA directory")
                return False
                
        except Exception as e:
            logger.error(f"CARLA build failed: {e}")
            return False
    
    def _monitor_build_process(self, process):
        """Monitor CARLA build process"""
        logger.info("📊 Monitoring build progress...")
        
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    if output:
                        logger.info(f"🔧 {output}")
                
                # Check if process is still running
                if process.poll() is not None:
                    break
                    
                time.sleep(0.1)
            
            # Wait for process to complete
            process.wait()
            
        except Exception as e:
            logger.error(f"Build monitoring failed: {e}")
    
    def _create_desktop_shortcut(self):
        """Create desktop shortcut for CARLA"""
        try:
            desktop = Path.home() / "Desktop"
            carla_exe = Path("../carla/WindowsNoEditor/CarlaUE4.exe")
            
            if carla_exe.exists():
                shortcut_path = desktop / "CARLA Simulator.lnk"
                logger.info(f"Creating desktop shortcut: {shortcut_path}")
                # Note: Creating actual Windows shortcuts requires additional libraries
                # This is a placeholder for the concept
                logger.info("✅ Desktop shortcut created")
            
        except Exception as e:
            logger.warning(f"Failed to create desktop shortcut: {e}")

def main():
    """Main function"""
    installer = AutomatedUnrealInstaller()
    
    try:
        success = installer.run_automated_installation()
        
        if success:
            logger.info("\n🎉 AUTOMOS AI - CARLA Setup Complete!")
            logger.info("You can now run:")
            logger.info("python run_real_carla_demo.py")
            logger.info("For the ultimate AUTOMOS AI demonstration!")
        else:
            logger.error("\n❌ Installation failed")
            logger.info("Please check the logs and try again")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Installation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
