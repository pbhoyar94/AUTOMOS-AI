#!/usr/bin/env python3
"""
Unreal Engine Manual Installation Guide
Step-by-step guide with visual instructions
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnrealEngineManualGuide:
    """Manual Unreal Engine installation guide"""
    
    def __init__(self):
        """Initialize guide"""
        self.installation_steps = [
            "Open Epic Games Launcher",
            "Navigate to Unreal Engine tab", 
            "Select Unreal Engine 4.27",
            "Click Install button",
            "Choose installation location",
            "Wait for download and installation",
            "Verify installation"
        ]
    
    def run_manual_installation_guide(self) -> bool:
        """Run comprehensive manual installation guide"""
        logger.info("🎮 Unreal Engine 4.27 Manual Installation Guide")
        logger.info("=" * 60)
        
        try:
            # Step 1: Verify Epic Games Launcher
            if not self._verify_epic_launcher():
                return self._install_epic_launcher()
            
            # Step 2: Show detailed installation steps
            self._show_detailed_installation_steps()
            
            # Step 3: Monitor installation progress
            if self._monitor_installation_progress():
                # Step 4: Verify installation
                if self._verify_installation():
                    # Step 5: Build CARLA
                    return self._build_carla()
            
            return False
            
        except Exception as e:
            logger.error(f"Manual guide failed: {e}")
            return False
    
    def _verify_epic_launcher(self) -> bool:
        """Verify Epic Games Launcher is installed"""
        logger.info("🔍 Checking Epic Games Launcher...")
        
        epic_paths = [
            Path("C:/Program Files/Epic Games/Launcher/Portal/Binaries/Win64/EpicGamesLauncher.exe"),
            Path("C:/Program Files (x86)/Epic Games/Launcher/Portal/Binaries/Win64/EpicGamesLauncher.exe")
        ]
        
        for path in epic_paths:
            if path.exists():
                logger.info(f"✅ Epic Games Launcher found: {path}")
                return True
        
        logger.error("❌ Epic Games Launcher not found")
        return False
    
    def _install_epic_launcher(self) -> bool:
        """Install Epic Games Launcher"""
        logger.info("📦 Installing Epic Games Launcher...")
        
        try:
            # Install Epic Games Launcher
            result = subprocess.run(['winget', 'install', 'EpicGames.EpicGamesLauncher'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Epic Games Launcher installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install Epic Games Launcher: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Epic Games Launcher installation failed: {e}")
            return False
    
    def _show_detailed_installation_steps(self):
        """Show detailed installation steps"""
        logger.info("\n" + "="*60)
        logger.info("DETAILED UNREAL ENGINE INSTALLATION STEPS")
        logger.info("="*60)
        
        logger.info("\n📋 STEP 1: OPEN EPIC GAMES LAUNCHER")
        logger.info("• Epic Games Launcher should be running")
        logger.info("• If not, find it in Start Menu or Desktop")
        logger.info("• Look for the Epic Games icon")
        
        logger.info("\n📋 STEP 2: NAVIGATE TO UNREAL ENGINE")
        logger.info("• In Epic Games Launcher, look at left sidebar")
        logger.info("• Click on 'Unreal Engine' tab")
        logger.info("• You'll see Unreal Engine versions available")
        
        logger.info("\n📋 STEP 3: SELECT UNREAL ENGINE 4.27")
        logger.info("• Find 'Unreal Engine 4.27' in the list")
        logger.info("• If 4.27 not available, use latest 4.x version")
        logger.info("• Click on the version to select it")
        
        logger.info("\n📋 STEP 4: CLICK INSTALL BUTTON")
        logger.info("• Look for 'Install' button on the right side")
        logger.info("• Click the button to start installation")
        logger.info("• Accept any terms if prompted")
        
        logger.info("\n📋 STEP 5: CHOOSE INSTALLATION LOCATION")
        logger.info("• Select installation folder:")
        logger.info("  - Recommended: C:/Program Files/Epic Games/UE_4.27")
        logger.info("  - Alternative: Any location with 20GB+ space")
        logger.info("• Click 'Install' to confirm location")
        
        logger.info("\n📋 STEP 6: WAIT FOR DOWNLOAD AND INSTALLATION")
        logger.info("• Download time: 20-40 minutes")
        logger.info("• Installation time: 10-20 minutes")
        logger.info("• Total time: 30-60 minutes")
        logger.info("• Monitor progress in Epic Games Launcher")
        
        logger.info("\n📋 STEP 7: VERIFY INSTALLATION")
        logger.info("• Wait for 'Installation Complete' message")
        logger.info("• Check that UE 4.27 appears in your library")
        logger.info("• Return to this script when complete")
        
        logger.info("\n" + "="*60)
        logger.info("⏳ INSTALLATION PROGRESS MONITOR")
        logger.info("="*60)
        logger.info("Complete the installation in Epic Games Launcher")
        logger.info("The script will check for installation every 30 seconds")
        logger.info("Press Ctrl+C to stop monitoring")
        
        return True
    
    def _monitor_installation_progress(self) -> bool:
        """Monitor installation progress"""
        logger.info("🔍 Starting installation monitoring...")
        
        check_count = 0
        max_checks = 120  # 1 hour maximum (120 * 30 seconds)
        
        try:
            while check_count < max_checks:
                check_count += 1
                
                # Check if UE is installed
                if self._verify_installation():
                    logger.info("✅ Unreal Engine installation detected!")
                    return True
                
                # Show progress
                minutes_elapsed = (check_count * 30) // 60
                logger.info(f"🔍 Checking installation... ({minutes_elapsed} minutes elapsed)")
                
                # Wait 30 seconds before next check
                time.sleep(30)
            
            logger.error("❌ Installation timeout reached (1 hour)")
            return False
            
        except KeyboardInterrupt:
            logger.info("\n⏹️ Installation monitoring stopped by user")
            return self._verify_installation()
    
    def _verify_installation(self) -> bool:
        """Verify Unreal Engine installation"""
        logger.info("🔍 Verifying Unreal Engine installation...")
        
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
                    logger.info(f"✅ Unreal Engine found: {ue_path}")
                    logger.info(f"✅ UE4Editor.exe found: {ue4_exe}")
                    return True
        
        logger.error("❌ Unreal Engine not found")
        logger.info("Please ensure Unreal Engine 4.27 is completely installed")
        return False
    
    def _build_carla(self) -> bool:
        """Build CARLA with Unreal Engine"""
        logger.info("🔧 Building CARLA with Unreal Engine...")
        
        try:
            carla_path = Path(__file__).parent.parent / "carla"
            
            if not carla_path.exists():
                logger.error("❌ CARLA directory not found")
                return False
            
            update_script = carla_path / "Update.bat"
            
            if not update_script.exists():
                logger.error("❌ Update.bat not found in CARLA directory")
                return False
            
            logger.info("🚀 Starting CARLA build...")
            logger.info("This will take 30-60 minutes")
            logger.info("Please be patient...")
            
            # Start CARLA build
            result = subprocess.run([str(update_script)], 
                                  cwd=str(carla_path),
                                  capture_output=True, 
                                  text=True,
                                  timeout=7200)  # 2 hour timeout
            
            if result.returncode == 0:
                logger.info("✅ CARLA build completed successfully")
                return True
            else:
                logger.error(f"❌ CARLA build failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ CARLA build timed out (2 hours)")
            return False
        except Exception as e:
            logger.error(f"CARLA build failed: {e}")
            return False
    
    def _show_next_steps(self):
        """Show next steps after successful installation"""
        logger.info("\n" + "="*60)
        logger.info("🎉 INSTALLATION COMPLETE - NEXT STEPS")
        logger.info("="*60)
        
        logger.info("\n🚀 RUN AUTOMOS AI ON REAL CARLA:")
        logger.info("python run_real_carla_demo.py")
        
        logger.info("\n🎬 ALTERNATIVE DEMOS:")
        logger.info("python run_carla_product_demo.py")
        logger.info("python run_extended_carla_demo.py")
        
        logger.info("\n📊 PERFORMANCE EXPECTATIONS:")
        logger.info("• Real 3D graphics with Unreal Engine")
        logger.info("• 30-60 FPS performance")
        logger.info("• Real physics simulation")
        logger.info("• Actual sensor data")
        logger.info("• Production-ready demonstrations")
        
        logger.info("\n🎯 FEATURES AVAILABLE:")
        logger.info("• Real autonomous driving")
        logger.info("• Professional visualization")
        logger.info("• Customer-ready demos")
        logger.info("• Industry-standard simulation")

def main():
    """Main function"""
    guide = UnrealEngineManualGuide()
    
    try:
        success = guide.run_manual_installation_guide()
        
        if success:
            logger.info("\n🎉 UNREAL ENGINE INSTALLATION COMPLETE!")
            logger.info("AUTOMOS AI is ready for real CARLA demonstrations!")
        else:
            logger.error("\n❌ Installation incomplete")
            logger.info("Please follow the steps and try again")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Installation guide stopped by user")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
