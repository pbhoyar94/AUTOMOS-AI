#!/usr/bin/env python3
"""
CARLA Status Checker
Check CARLA installation and server availability
"""

import os
import sys
import socket
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_carla_installation():
    """Check if CARLA is properly installed"""
    logger.info("Checking CARLA installation...")
    
    carla_root = Path(__file__).parent.parent / "carla"
    
    if not carla_root.exists():
        logger.error("❌ CARLA directory not found")
        return False
    
    logger.info(f"✅ CARLA directory found: {carla_root}")
    
    # Check for CARLA executable
    carla_executable_paths = [
        carla_root / "WindowsNoEditor" / "CarlaUE4.exe",
        carla_root / "CarlaUE4.exe",
        carla_root / "Binaries" / "Win64" / "CarlaUE4.exe",
        carla_root / "LinuxNoEditor" / "CarlaUE4.sh",
        carla_root / "CarlaUE4.sh",
        carla_root / "Binaries" / "Linux" / "CarlaUE4.sh"
    ]
    
    carla_executable = None
    for path in carla_executable_paths:
        if path.exists():
            carla_executable = path
            break
    
    if carla_executable:
        logger.info(f"✅ CARLA executable found: {carla_executable}")
        return True
    else:
        logger.error("❌ CARLA executable not found")
        logger.info("CARLA needs to be built or pre-built binaries downloaded")
        return False

def check_carla_python_api():
    """Check if CARLA Python API is available"""
    logger.info("Checking CARLA Python API...")
    
    try:
        import carla
        logger.info("✅ CARLA Python API is available")
        return True
    except ImportError as e:
        logger.error(f"❌ CARLA Python API not available: {e}")
        return False

def check_carla_server(host='localhost', port=2000):
    """Check if CARLA server is running"""
    logger.info(f"Checking CARLA server at {host}:{port}...")
    
    try:
        # Try to connect to CARLA server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            logger.info(f"✅ CARLA server is running at {host}:{port}")
            return True
        else:
            logger.error(f"❌ CARLA server is not running at {host}:{port}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking CARLA server: {e}")
        return False

def check_unreal_engine():
    """Check if Unreal Engine is available for building CARLA"""
    logger.info("Checking Unreal Engine availability...")
    
    # Check common Unreal Engine installation paths
    ue_paths = [
        Path("C:/Program Files/Epic Games/UE_4.27"),
        Path("C:/Program Files/Epic Games/UE_4.26"),
        Path("C:/Program Files/Epic Games/UE_4.25"),
        Path("C:/Program Files (x86)/Epic Games/UE_4.27"),
        Path("C:/Program Files (x86)/Epic Games/UE_4.26"),
        Path("C:/Program Files (x86)/Epic Games/UE_4.25"),
        Path.home() / "UnrealEngine" / "UE_4.27",
        Path.home() / "UnrealEngine" / "UE_4.26",
        Path.home() / "UnrealEngine" / "UE_4.25"
    ]
    
    for ue_path in ue_paths:
        if ue_path.exists():
            logger.info(f"✅ Unreal Engine found: {ue_path}")
            return True
    
    logger.error("❌ Unreal Engine not found")
    logger.info("Unreal Engine 4.24+ is required to build CARLA from source")
    return False

def check_system_requirements():
    """Check system requirements for CARLA"""
    logger.info("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 7):
        logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        logger.error(f"❌ Python version too old: {python_version.major}.{python_version.minor}.{python_version.micro}")
        return False
    
    # Check available memory (Windows)
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 16:
            logger.info(f"✅ Available RAM: {memory_gb:.1f} GB")
        else:
            logger.warning(f"⚠️ Low RAM: {memory_gb:.1f} GB (16GB+ recommended)")
    except ImportError:
        logger.warning("⚠️ Cannot check RAM (psutil not installed)")
    
    # Check GPU (basic check)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"✅ GPU available: {gpu_count} CUDA device(s)")
        else:
            logger.warning("⚠️ No CUDA GPU available (CPU-only mode)")
    except ImportError:
        logger.warning("⚠️ Cannot check GPU (PyTorch not installed)")
    
    return True

def suggest_next_steps():
    """Suggest next steps based on current status"""
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDED NEXT STEPS")
    logger.info("="*60)
    
    carla_installed = check_carla_installation()
    carla_api_available = check_carla_python_api()
    carla_server_running = check_carla_server()
    ue_available = check_unreal_engine()
    
    if not carla_installed:
        logger.info("\n1. INSTALL CARLA:")
        logger.info("   Option A: Download pre-built binaries")
        logger.info("   Option B: Build from source (requires Unreal Engine)")
        logger.info("   Run: python start_carla_server.py --download")
    
    elif not carla_api_available:
        logger.info("\n2. INSTALL CARLA PYTHON API:")
        logger.info("   Run: cd ../carla/PythonAPI/carla && python setup.py install")
        logger.info("   Or: pip install carla (if available)")
    
    elif not carla_server_running:
        logger.info("\n3. START CARLA SERVER:")
        logger.info("   Run: python start_carla_server.py")
        logger.info("   Or manually start CARLA executable")
    
    else:
        logger.info("\n✅ CARLA IS READY!")
        logger.info("   Run: python carla_integration.py")
        logger.info("   Run: python test_complete_carla.py")

def main():
    """Main function"""
    logger.info("CARLA Status Check")
    logger.info("="*60)
    
    # Check all components
    checks = [
        ("System Requirements", check_system_requirements),
        ("CARLA Installation", check_carla_installation),
        ("CARLA Python API", check_carla_python_api),
        ("CARLA Server", check_carla_server),
        ("Unreal Engine", check_unreal_engine)
    ]
    
    results = {}
    for check_name, check_func in checks:
        logger.info(f"\n{check_name}:")
        results[check_name] = check_func()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{check_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} checks passed")
    
    # Suggest next steps
    suggest_next_steps()
    
    return 0 if passed >= 3 else 1  # At least 3 checks should pass

if __name__ == "__main__":
    sys.exit(main())
