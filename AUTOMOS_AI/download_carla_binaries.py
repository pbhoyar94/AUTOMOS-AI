#!/usr/bin/env python3
"""
CARLA Binaries Downloader
Download CARLA pre-built binaries for Windows
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_carla_windows():
    """Download CARLA Windows binaries"""
    carla_root = Path(__file__).parent.parent / "carla"
    
    # CARLA 0.9.15 Windows download URL
    carla_url = "https://carla-releases.s3.amazonaws.com/Windows/Dev/CARLA_0.9.15.zip"
    additional_maps_url = "https://carla-releases.s3.amazonaws.com/Windows/Dev/AdditionalMaps_0.9.15.zip"
    
    try:
        logger.info("Downloading CARLA 0.9.15 Windows binaries...")
        
        # Download CARLA main package
        carla_zip_path = carla_root / "CARLA_0.9.15.zip"
        
        if not carla_zip_path.exists():
            logger.info(f"Downloading from: {carla_url}")
            urllib.request.urlretrieve(carla_url, carla_zip_path)
            logger.info("CARLA download completed")
        else:
            logger.info("CARLA zip already exists, skipping download")
        
        # Extract CARLA
        if not (carla_root / "WindowsNoEditor").exists():
            logger.info("Extracting CARLA...")
            with zipfile.ZipFile(carla_zip_path, 'r') as zip_ref:
                zip_ref.extractall(carla_root)
            logger.info("CARLA extraction completed")
        else:
            logger.info("CARLA already extracted, skipping extraction")
        
        # Download additional maps (optional)
        additional_maps_zip_path = carla_root / "AdditionalMaps_0.9.15.zip"
        
        if not additional_maps_zip_path.exists():
            logger.info("Downloading additional maps...")
            urllib.request.urlretrieve(additional_maps_url, additional_maps_zip_path)
            logger.info("Additional maps download completed")
        else:
            logger.info("Additional maps zip already exists, skipping download")
        
        # Extract additional maps
        if not (carla_root / "Import").exists():
            logger.info("Extracting additional maps...")
            with zipfile.ZipFile(additional_maps_zip_path, 'r') as zip_ref:
                zip_ref.extractall(carla_root)
            logger.info("Additional maps extraction completed")
        else:
            logger.info("Additional maps already extracted, skipping extraction")
        
        # Clean up zip files
        if carla_zip_path.exists():
            carla_zip_path.unlink()
            logger.info("Cleaned up CARLA zip file")
        
        if additional_maps_zip_path.exists():
            additional_maps_zip_path.unlink()
            logger.info("Cleaned up additional maps zip file")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download CARLA: {e}")
        return False

def install_carla_python_api():
    """Install CARLA Python API"""
    carla_root = Path(__file__).parent.parent / "carla"
    python_api_path = carla_root / "WindowsNoEditor" / "PythonAPI" / "carla"
    
    if not python_api_path.exists():
        logger.error("CARLA Python API not found. Please download CARLA first.")
        return False
    
    try:
        logger.info("Installing CARLA Python API...")
        
        # Add CARLA Python API to path
        sys.path.insert(0, str(python_api_path))
        
        # Try to import and install
        os.chdir(python_api_path)
        
        # Install dependencies
        os.system("pip install -r requirements.txt")
        
        # Install CARLA
        os.system("python setup.py install")
        
        logger.info("CARLA Python API installation completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to install CARLA Python API: {e}")
        return False

def test_carla_installation():
    """Test CARLA installation"""
    try:
        logger.info("Testing CARLA installation...")
        
        # Test Python API
        import carla
        logger.info("✅ CARLA Python API imported successfully")
        
        # Test connection (will fail if server not running, but that's expected)
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(2.0)
            world = client.get_world()
            logger.info("✅ CARLA server connection successful")
        except Exception:
            logger.info("ℹ️ CARLA server not running (this is expected)")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ CARLA installation test failed: {e}")
        return False

def main():
    """Main function"""
    logger.info("CARLA Binaries Downloader")
    logger.info("="*50)
    
    # Download CARLA
    if download_carla_windows():
        logger.info("✅ CARLA download completed successfully")
    else:
        logger.error("❌ CARLA download failed")
        return 1
    
    # Install Python API
    if install_carla_python_api():
        logger.info("✅ CARLA Python API installed successfully")
    else:
        logger.error("❌ CARLA Python API installation failed")
        return 1
    
    # Test installation
    if test_carla_installation():
        logger.info("✅ CARLA installation test passed")
    else:
        logger.error("❌ CARLA installation test failed")
        return 1
    
    logger.info("="*50)
    logger.info("🎉 CARLA installation completed successfully!")
    logger.info("You can now start CARLA server with:")
    logger.info("python start_carla_server.py")
    logger.info("Or run CARLA directly:")
    logger.info("cd ../carla/WindowsNoEditor && CarlaUE4.exe -windowed -carla-server")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
