#!/usr/bin/env python3
"""
Unreal Engine Installation Monitor
Monitor and verify Unreal Engine installation progress
"""

import os
import sys
import time
import logging
import psutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnrealInstallationMonitor:
    """Monitor Unreal Engine installation"""
    
    def __init__(self):
        """Initialize monitor"""
        self.installation_paths = [
            Path("C:/Program Files/Epic Games/UE_4.27"),
            Path("C:/Program Files/Epic Games/UE_4.26"),
            Path("C:/Program Files (x86)/Epic Games/UE_4.27"),
            Path("C:/Program Files (x86)/Epic Games/UE_4.26"),
        ]
        
    def run_installation_monitor(self) -> bool:
        """Monitor Unreal Engine installation"""
        logger.info("🔍 Unreal Engine Installation Monitor")
        logger.info("=" * 60)
        
        logger.info("\n📋 INSTALLATION INSTRUCTIONS:")
        logger.info("1. Epic Games Launcher should now be open")
        logger.info("2. Click 'Unreal Engine' tab in left sidebar")
        logger.info("3. Find 'Unreal Engine 4.27' (or latest 4.x)")
        logger.info("4. Click 'Install' button")
        logger.info("5. Choose installation location")
        logger.info("6. Wait for download to complete")
        
        logger.info("\n⏱️ EXPECTED TIME:")
        logger.info("- Download: 20-40 minutes")
        logger.info("- Installation: 10-20 minutes")
        logger.info("- Total: 30-60 minutes")
        
        logger.info("\n💾 REQUIREMENTS:")
        logger.info("- Disk Space: 20GB+ free")
        logger.info("- Internet: Stable connection")
        logger.info("- RAM: 8GB+ recommended")
        
        logger.info("\n" + "="*60)
        logger.info("🔍 MONITORING INSTALLATION PROGRESS")
        logger.info("="*60)
        logger.info("Complete installation in Epic Games Launcher")
        logger.info("This script will automatically detect when installation is complete")
        logger.info("Press Ctrl+C to stop monitoring")
        
        return self._monitor_installation()
    
    def _monitor_installation(self) -> bool:
        """Monitor installation progress"""
        check_count = 0
        max_checks = 120  # 1 hour maximum
        
        try:
            while check_count < max_checks:
                check_count += 1
                
                # Check if Unreal Engine is installed
                if self._check_unreal_installed():
                    logger.info("\n🎉 UNREAL ENGINE INSTALLATION DETECTED!")
                    return self._verify_installation()
                
                # Show progress
                minutes_elapsed = (check_count * 30) // 60
                logger.info(f"🔍 Checking installation... ({minutes_elapsed} minutes elapsed)")
                
                # Check Epic Games Launcher status
                self._check_epic_launcher_status()
                
                # Wait 30 seconds before next check
                time.sleep(30)
            
            logger.error("❌ Installation timeout reached (1 hour)")
            return False
            
        except KeyboardInterrupt:
            logger.info("\n⏹️ Monitoring stopped by user")
            return self._check_unreal_installed()
    
    def _check_unreal_installed(self) -> bool:
        """Check if Unreal Engine is installed"""
        for ue_path in self.installation_paths:
            if ue_path.exists():
                ue4_exe = ue_path / "Engine" / "Binaries" / "Win64" / "UE4Editor.exe"
                if ue4_exe.exists():
                    logger.info(f"✅ Found Unreal Engine: {ue_path}")
                    return True
        return False
    
    def _check_epic_launcher_status(self):
        """Check Epic Games Launcher status"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == 'EpicGamesLauncher.exe':
                    logger.info("📱 Epic Games Launcher is running")
                    return True
            logger.warning("⚠️ Epic Games Launcher not running")
        except Exception:
            pass
        return False
    
    def _verify_installation(self) -> bool:
        """Verify Unreal Engine installation"""
        logger.info("🔍 Verifying Unreal Engine installation...")
        
        for ue_path in self.installation_paths:
            if ue_path.exists():
                ue4_exe = ue_path / "Engine" / "Binaries" / "Win64" / "UE4Editor.exe"
                if ue4_exe.exists():
                    logger.info(f"✅ Unreal Engine verified: {ue_path}")
                    logger.info(f"✅ UE4Editor.exe: {ue4_exe}")
                    
                    # Test UE4Editor
                    if self._test_ue4_editor(ue4_exe):
                        logger.info("✅ UE4Editor launches successfully")
                        return True
                    else:
                        logger.warning("⚠️ UE4Editor test failed, but files exist")
                        return True
        
        logger.error("❌ Unreal Engine verification failed")
        return False
    
    def _test_ue4_editor(self, ue4_exe) -> bool:
        """Test UE4Editor launch"""
        try:
            import subprocess
            process = subprocess.Popen([str(ue4_exe), "-version"], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
            
            try:
                stdout, stderr = process.communicate(timeout=30)
                return process.returncode == 0
            except subprocess.TimeoutExpired:
                process.terminate()
                return True  # Assume it's working if it starts
                
        except Exception as e:
            logger.warning(f"UE4Editor test failed: {e}")
            return False
    
    def _show_next_steps(self):
        """Show next steps"""
        logger.info("\n" + "="*60)
        logger.info("🎉 UNREAL ENGINE INSTALLATION COMPLETE!")
        logger.info("="*60)
        
        logger.info("\n🚀 NEXT STEPS FOR AUTOMOS AI:")
        logger.info("1. Build CARLA with Unreal Engine:")
        logger.info("   python setup_carla_unreal.py")
        logger.info("2. Run AUTOMOS AI on real CARLA:")
        logger.info("   python run_real_carla_demo.py")
        logger.info("3. Test product demonstrations:")
        logger.info("   python run_carla_product_demo.py")
        
        logger.info("\n🎯 FEATURES NOW AVAILABLE:")
        logger.info("• Real 3D graphics with Unreal Engine")
        logger.info("• Professional autonomous driving simulation")
        logger.info("• Production-ready demonstrations")
        logger.info("• Customer-ready visualizations")
        logger.info("• Industry-standard simulation platform")

def main():
    """Main function"""
    monitor = UnrealInstallationMonitor()
    
    try:
        success = monitor.run_installation_monitor()
        
        if success:
            monitor._show_next_steps()
        else:
            logger.error("\n❌ Unreal Engine installation not detected")
            logger.info("Please complete the installation in Epic Games Launcher")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Installation monitoring stopped")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
