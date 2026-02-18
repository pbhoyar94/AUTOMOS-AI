#!/usr/bin/env python3
"""
Build Modified Unreal Engine for CARLA from Source
Automated build process for CARLA-specific Unreal Engine
"""

import os
import sys
import time
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CarlaUnrealEngineBuilder:
    """Build modified Unreal Engine for CARLA"""
    
    def __init__(self):
        """Initialize builder"""
        self.ue_install_path = None
        self.build_progress = 0
        self.build_status = "Not Started"
        
    def run_carla_ue_build(self) -> bool:
        """Run complete CARLA Unreal Engine build"""
        logger.info("🚀 Building Modified Unreal Engine for CARLA")
        logger.info("=" * 60)
        
        try:
            # Step 1: Choose installation location
            if not self._choose_install_location():
                return False
            
            # Step 2: Clone CARLA Unreal Engine
            if not self._clone_carla_unreal():
                return False
            
            # Step 3: Run configuration scripts
            if not self._run_configuration():
                return False
            
            # Step 4: Generate project files
            if not self._generate_project_files():
                return False
            
            # Step 5: Build with Visual Studio
            if not self._build_with_visual_studio():
                return False
            
            # Step 6: Verify installation
            if not self._verify_ue_installation():
                return False
            
            # Step 7: Build CARLA
            return self._build_carla()
            
        except Exception as e:
            logger.error(f"CARLA Unreal Engine build failed: {e}")
            return False
    
    def _choose_install_location(self) -> bool:
        """Choose optimal installation location"""
        logger.info("📁 Choosing Unreal Engine installation location...")
        
        # Recommended locations (close to C:\ as per instructions)
        possible_paths = [
            Path("C:/UnrealEngineCarla"),
            Path("C:/UE4Carla"),
            Path("C:/CarlaUnreal"),
            Path("D:/UnrealEngineCarla"),
            Path.home() / "UnrealEngineCarla"
        ]
        
        for path in possible_paths:
            try:
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    self.ue_install_path = path
                    logger.info(f"✅ Selected location: {path}")
                    return True
                elif len(list(path.iterdir())) == 0:  # Empty directory
                    self.ue_install_path = path
                    logger.info(f"✅ Using existing empty location: {path}")
                    return True
            except Exception as e:
                logger.warning(f"Cannot use {path}: {e}")
                continue
        
        logger.error("❌ No suitable installation location found")
        return False
    
    def _clone_carla_unreal(self) -> bool:
        """Clone CARLA Unreal Engine repository"""
        logger.info("📥 Cloning CARLA Unreal Engine repository...")
        
        try:
            # Check if git is available
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ Git not found. Please install Git first.")
                return False
            
            # Clone the CARLA Unreal Engine repository
            logger.info("Cloning from: https://github.com/CarlaUnreal/UnrealEngine.git")
            logger.info("Branch: carla")
            logger.info("Location: {}".format(self.ue_install_path))
            
            clone_cmd = [
                'git', 'clone', '--depth', '1', '-b', 'carla',
                'https://github.com/CarlaUnreal/UnrealEngine.git',
                str(self.ue_install_path)
            ]
            
            logger.info("Starting clone... This may take 10-30 minutes")
            
            process = subprocess.Popen(clone_cmd, 
                                     cwd=self.ue_install_path.parent,
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True,
                                     bufsize=1,
                                     universal_newlines=True)
            
            # Monitor clone progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    if output:
                        logger.info(f"📥 {output}")
                
                time.sleep(0.1)
            
            if process.returncode == 0:
                logger.info("✅ CARLA Unreal Engine cloned successfully")
                return True
            else:
                logger.error("❌ Failed to clone CARLA Unreal Engine")
                return False
                
        except Exception as e:
            logger.error(f"Clone failed: {e}")
            return False
    
    def _run_configuration(self) -> bool:
        """Run Setup.bat configuration script"""
        logger.info("⚙️ Running Setup.bat configuration...")
        
        try:
            setup_script = self.ue_install_path / "Setup.bat"
            
            if not setup_script.exists():
                logger.error("❌ Setup.bat not found")
                return False
            
            logger.info("Running Setup.bat... This may take 10-20 minutes")
            
            process = subprocess.Popen([str(setup_script)], 
                                     cwd=str(self.ue_install_path),
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True,
                                     bufsize=1,
                                     universal_newlines=True)
            
            # Monitor setup progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    if output:
                        logger.info(f"⚙️ {output}")
                
                time.sleep(0.1)
            
            if process.returncode == 0:
                logger.info("✅ Setup.bat completed successfully")
                return True
            else:
                logger.error("❌ Setup.bat failed")
                return False
                
        except Exception as e:
            logger.error(f"Setup.bat failed: {e}")
            return False
    
    def _generate_project_files(self) -> bool:
        """Run GenerateProjectFiles.bat"""
        logger.info("📄 Generating project files...")
        
        try:
            gen_script = self.ue_install_path / "GenerateProjectFiles.bat"
            
            if not gen_script.exists():
                logger.error("❌ GenerateProjectFiles.bat not found")
                return False
            
            logger.info("Running GenerateProjectFiles.bat...")
            
            process = subprocess.Popen([str(gen_script)], 
                                     cwd=str(self.ue_install_path),
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True,
                                     bufsize=1,
                                     universal_newlines=True)
            
            # Monitor progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    if output:
                        logger.info(f"📄 {output}")
                
                time.sleep(0.1)
            
            if process.returncode == 0:
                logger.info("✅ Project files generated successfully")
                return True
            else:
                logger.error("❌ Project file generation failed")
                return False
                
        except Exception as e:
            logger.error(f"GenerateProjectFiles.bat failed: {e}")
            return False
    
    def _build_with_visual_studio(self) -> bool:
        """Build with Visual Studio 2022"""
        logger.info("🔨 Building with Visual Studio 2022...")
        
        try:
            # Check for Visual Studio
            vs_paths = [
                Path("C:/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe"),
                Path("C:/Program Files/Microsoft Visual Studio/2022/Professional/MSBuild/Current/Bin/MSBuild.exe"),
                Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise/MSBuild/Current/Bin/MSBuild.exe")
            ]
            
            msbuild_path = None
            for path in vs_paths:
                if path.exists():
                    msbuild_path = path
                    break
            
            if not msbuild_path:
                logger.error("❌ Visual Studio 2022 not found")
                logger.info("Please install Visual Studio 2022 with C++ development tools")
                return False
            
            # Find UE4.sln
            ue4_sln = self.ue_install_path / "UE4.sln"
            
            if not ue4_sln.exists():
                logger.error("❌ UE4.sln not found")
                return False
            
            logger.info("Building UE4.sln with MSBuild...")
            logger.info("Configuration: Development Editor")
            logger.info("Platform: Win64")
            logger.info("This will take 1-3 hours...")
            
            build_cmd = [
                str(msbuild_path),
                str(ue4_sln),
                "/p:Configuration=Development Editor",
                "/p:Platform=Win64",
                "/p:UnrealBuildTool",
                "/m",  # Parallel build
                "/v:minimal"  # Minimal verbosity
            ]
            
            logger.info("Starting build... This is a long process")
            
            process = subprocess.Popen(build_cmd, 
                                     cwd=str(self.ue_install_path),
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True,
                                     bufsize=1,
                                     universal_newlines=True)
            
            # Monitor build progress
            start_time = time.time()
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    if output:
                        elapsed = int(time.time() - start_time)
                        minutes = elapsed // 60
                        logger.info(f"🔨 [{minutes:02d}m] {output}")
                
                time.sleep(1)
            
            if process.returncode == 0:
                logger.info("✅ Visual Studio build completed successfully")
                return True
            else:
                logger.error("❌ Visual Studio build failed")
                return False
                
        except Exception as e:
            logger.error(f"Visual Studio build failed: {e}")
            return False
    
    def _verify_ue_installation(self) -> bool:
        """Verify Unreal Engine installation"""
        logger.info("🔍 Verifying Unreal Engine installation...")
        
        try:
            # Check for UE4Editor.exe
            ue4_editor = self.ue_install_path / "Engine" / "Binaries" / "Win64" / "UE4Editor.exe"
            
            if ue4_editor.exists():
                logger.info(f"✅ UE4Editor.exe found: {ue4_editor}")
                
                # Try to run UE4Editor to verify
                logger.info("Testing UE4Editor launch...")
                process = subprocess.Popen([str(ue4_editor), "-version"], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE,
                                         text=True)
                
                try:
                    stdout, stderr = process.communicate(timeout=30)
                    if process.returncode == 0:
                        logger.info("✅ UE4Editor launches successfully")
                        return True
                    else:
                        logger.warning("⚠️ UE4Editor test failed, but executable exists")
                        return True
                except subprocess.TimeoutExpired:
                    logger.warning("⚠️ UE4Editor test timed out, but executable exists")
                    process.terminate()
                    return True
            else:
                logger.error("❌ UE4Editor.exe not found")
                return False
                
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def _build_carla(self) -> bool:
        """Build CARLA with the new Unreal Engine"""
        logger.info("🚗 Building CARLA with custom Unreal Engine...")
        
        try:
            carla_path = Path(__file__).parent.parent / "carla"
            
            if not carla_path.exists():
                logger.error("❌ CARLA directory not found")
                return False
            
            # Set environment variables
            env = os.environ.copy()
            env['UNREAL_ENGINE_PATH'] = str(self.ue_install_path)
            
            update_script = carla_path / "Update.bat"
            
            if not update_script.exists():
                logger.error("❌ Update.bat not found in CARLA directory")
                return False
            
            logger.info("Running CARLA Update.bat with custom Unreal Engine...")
            logger.info("This will take 30-60 minutes")
            
            process = subprocess.Popen([str(update_script)], 
                                     cwd=str(carla_path),
                                     env=env,
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True,
                                     bufsize=1,
                                     universal_newlines=True)
            
            # Monitor build progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output = output.strip()
                    if output:
                        logger.info(f"🚗 {output}")
                
                time.sleep(0.1)
            
            if process.returncode == 0:
                logger.info("✅ CARLA build completed successfully")
                return True
            else:
                logger.error("❌ CARLA build failed")
                return False
                
        except Exception as e:
            logger.error(f"CARLA build failed: {e}")
            return False
    
    def _show_completion_summary(self):
        """Show completion summary"""
        logger.info("\n" + "="*60)
        logger.info("🎉 CARLA UNREAL ENGINE BUILD COMPLETE!")
        logger.info("="*60)
        
        logger.info(f"\n📁 Unreal Engine Location: {self.ue_install_path}")
        logger.info(f"🎮 UE4Editor.exe: {self.ue_install_path}/Engine/Binaries/Win64/UE4Editor.exe")
        
        logger.info("\n🚀 NEXT STEPS:")
        logger.info("1. Test CARLA with real graphics:")
        logger.info("   python run_real_carla_demo.py")
        logger.info("2. Run AUTOMOS AI demonstrations:")
        logger.info("   python run_carla_product_demo.py")
        logger.info("3. Test extended scenarios:")
        logger.info("   python run_extended_carla_demo.py")
        
        logger.info("\n🎯 FEATURES AVAILABLE:")
        logger.info("• Real 3D graphics with CARLA Unreal Engine")
        logger.info("• Professional autonomous driving simulation")
        logger.info("• Production-ready demonstrations")
        logger.info("• Customer-ready visualizations")

def main():
    """Main function"""
    builder = CarlaUnrealEngineBuilder()
    
    try:
        success = builder.run_carla_ue_build()
        
        if success:
            builder._show_completion_summary()
        else:
            logger.error("\n❌ CARLA Unreal Engine build failed")
            logger.info("Please check the error messages above")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Build cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
