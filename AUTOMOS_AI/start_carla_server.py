#!/usr/bin/env python3
"""
CARLA Server Starter
Script to start CARLA server for AUTOMOS AI testing
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def start_carla_server():
    """Start CARLA server"""
    carla_root = Path(__file__).parent.parent / "carla"
    
    # Check if CARLA exists
    if not carla_root.exists():
        logger.error("CARLA not found. Please clone CARLA repository first.")
        return False
    
    # Look for CARLA executable
    carla_executable = None
    
    # Windows paths
    if os.name == 'nt':
        possible_paths = [
            carla_root / "WindowsNoEditor" / "CarlaUE4.exe",
            carla_root / "CarlaUE4.exe",
            carla_root / "Binaries" / "Win64" / "CarlaUE4.exe"
        ]
    # Linux paths
    else:
        possible_paths = [
            carla_root / "LinuxNoEditor" / "CarlaUE4.sh",
            carla_root / "CarlaUE4.sh",
            carla_root / "Binaries" / "Linux" / "CarlaUE4.sh"
        ]
    
    for path in possible_paths:
        if path.exists():
            carla_executable = path
            break
    
    if not carla_executable:
        logger.error("CARLA executable not found")
        logger.info("Please build CARLA first or download pre-built binaries")
        return False
    
    try:
        logger.info(f"Starting CARLA server: {carla_executable}")
        
        # CARLA server command line arguments
        cmd = [
            str(carla_executable),
            "-windowed",  # Run in windowed mode
            "-carla-server",  # Run as server
            "-world-port=2000",  # Server port
            "-quality-level=Low",  # Lower quality for better performance
            "-no-rendering-offscreen",  # Enable rendering
        ]
        
        # Start CARLA server
        process = subprocess.Popen(cmd, cwd=str(carla_root))
        
        logger.info("CARLA server starting...")
        logger.info("Wait for server to initialize (about 30 seconds)")
        
        # Wait a bit for server to start
        time.sleep(30)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("CARLA server started successfully")
            logger.info(f"Process ID: {process.pid}")
            return process
        else:
            logger.error("CARLA server failed to start")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start CARLA server: {e}")
        return False

def download_carla_binaries():
    """Download CARLA pre-built binaries"""
    import urllib.request
    import zipfile
    
    carla_root = Path(__file__).parent.parent / "carla"
    
    if os.name == 'nt':
        # Windows download URL
        carla_url = "http://carla-releases.s3.amazonaws.com/Windows/Dev/CARLA_Latest.zip"
        extract_to = carla_root
    else:
        # Linux download URL
        carla_url = "http://carla-releases.s3.amazonaws.com/Linux/Dev/CARLA_Latest.tar.gz"
        extract_to = carla_root.parent
    
    try:
        logger.info(f"Downloading CARLA from {carla_url}")
        
        # Download file
        filename = carla_url.split('/')[-1]
        download_path = carla_root / filename
        
        urllib.request.urlretrieve(carla_url, download_path)
        logger.info(f"Downloaded {filename}")
        
        # Extract
        if filename.endswith('.zip'):
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif filename.endswith('.tar.gz'):
            import tarfile
            with tarfile.open(download_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        
        logger.info("CARLA binaries extracted successfully")
        
        # Clean up download
        download_path.unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download CARLA: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start CARLA server for AUTOMOS AI")
    parser.add_argument("--download", action="store_true", help="Download CARLA binaries")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.download:
        logger.info("Downloading CARLA binaries...")
        if download_carla_binaries():
            logger.info("CARLA downloaded successfully")
        else:
            logger.error("Failed to download CARLA")
            return 1
    
    logger.info("Starting CARLA server...")
    process = start_carla_server()
    
    if process:
        try:
            # Keep script running
            logger.info("CARLA server running. Press Ctrl+C to stop.")
            process.wait()
        except KeyboardInterrupt:
            logger.info("Stopping CARLA server...")
            process.terminate()
            process.wait()
            logger.info("CARLA server stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
