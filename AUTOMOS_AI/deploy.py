#!/usr/bin/env python3
"""
AUTOMOS AI Deployment Script
Creates production deployment packages for customer deployment
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from edge_optimization.model_quantizer import ModelQuantizer
from edge_optimization.edge_optimizer import EdgeOptimizer
from edge_optimization.deployment_manager import DeploymentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_production_package(source_dir: str, output_dir: str, deployment_type: str = 'production'):
    """Create production deployment package"""
    
    logger.info(f"Creating {deployment_type} deployment package...")
    
    # Initialize components
    quantizer = ModelQuantizer()
    optimizer = EdgeOptimizer()
    deployment_manager = DeploymentManager()
    
    try:
        # Step 1: Quantize models
        logger.info("Step 1: Quantizing models...")
        models_dir = os.path.join(source_dir, 'models')
        if os.path.exists(models_dir):
            quantization_results = quantizer.quantize_all_models(models_dir, output_dir)
            logger.info(f"Quantization completed: {len(quantization_results)} models processed")
        
        # Step 2: Optimize for edge deployment
        logger.info("Step 2: Optimizing for edge deployment...")
        optimization_report = optimizer.optimize_for_edge({})
        
        # Step 3: Create deployment package
        logger.info("Step 3: Creating deployment package...")
        deployment_result = deployment_manager.create_deployment_package(
            source_directory=source_dir,
            output_directory=output_dir,
            deployment_type=deployment_type
        )
        
        if deployment_result['success']:
            logger.info("✅ Production package created successfully!")
            logger.info(f"📦 Package: {deployment_result['package_path']}")
            logger.info(f"📏 Size: {deployment_result['package_size_mb']:.1f} MB")
            logger.info(f"🔐 Checksum: {deployment_result['checksum']}")
            
            # Generate deployment script
            script_path = os.path.join(output_dir, 'deploy_automos_ai.sh')
            optimizer.generate_deployment_script(optimization_report, script_path)
            logger.info(f"📜 Deployment script: {script_path}")
            
            return True
        else:
            logger.error(f"❌ Failed to create deployment package: {deployment_result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return False

def main():
    """Main deployment function"""
    
    parser = argparse.ArgumentParser(description="AUTOMOS AI Deployment Tool")
    parser.add_argument("--source", type=str, default=str(PROJECT_ROOT),
                       help="Source directory containing AUTOMOS AI")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "deployments"),
                       help="Output directory for deployment packages")
    parser.add_argument("--type", choices=["development", "staging", "production"],
                       default="production", help="Deployment type")
    parser.add_argument("--verify", action="store_true",
                       help="Verify deployment package after creation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting AUTOMOS AI Deployment")
    logger.info(f"📂 Source: {args.source}")
    logger.info(f"📁 Output: {args.output}")
    logger.info(f"🏷️  Type: {args.type}")
    
    # Create deployment package
    success = create_production_package(args.source, str(output_dir), args.type)
    
    if success:
        logger.info("🎉 Deployment completed successfully!")
        
        # Verify package if requested
        if args.verify:
            logger.info("🔍 Verifying deployment package...")
            deployment_manager = DeploymentManager()
            
            # Find the latest package
            packages = list(output_dir.glob("automos_ai_deployment_*.zip"))
            if packages:
                latest_package = max(packages, key=os.path.getctime)
                verification = deployment_manager.verify_deployment(str(latest_package))
                
                if verification['success']:
                    logger.info("✅ Package verification passed!")
                    logger.info(f"📋 Files: {verification['file_count']}")
                    logger.info(f"🔐 Checksum: {verification['checksum']}")
                else:
                    logger.error(f"❌ Package verification failed: {verification['error']}")
            else:
                logger.warning("⚠️  No deployment package found for verification")
        
        return 0
    else:
        logger.error("💥 Deployment failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
