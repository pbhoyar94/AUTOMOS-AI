#!/usr/bin/env python3
"""
AUTOMOS AI Main Entry Point
World's First Reasoning-Based Autonomous Driving Engine
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.reasoning_engine import ReasoningEngine
from perception.perception_pipeline import PerceptionPipeline
from safety.safety_critic import SafetyCritic
from world_model.world_model import WorldModel
from integration.system_coordinator import SystemCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AUTOMOS_AI:
    """Main AUTOMOS AI System"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize AUTOMOS AI system
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing AUTOMOS AI System...")
        
        # Initialize core components
        self.reasoning_engine = ReasoningEngine()
        self.perception_pipeline = PerceptionPipeline()
        self.safety_critic = SafetyCritic()
        self.world_model = WorldModel()
        self.system_coordinator = SystemCoordinator()
        
        # System state
        self.is_running = False
        self.emergency_override = False
        
        logger.info("AUTOMOS AI System initialized successfully")
    
    def start(self, mode: str = "simulation", hardware: str = "cpu"):
        """
        Start the AUTOMOS AI system
        
        Args:
            mode: Operation mode (simulation/deployment)
            hardware: Target hardware (cpu/gpu/edge)
        """
        logger.info(f"Starting AUTOMOS AI in {mode} mode on {hardware}")
        
        try:
            # Initialize components based on mode
            self._initialize_components(mode, hardware)
            
            # Start system coordinator
            self.system_coordinator.start()
            
            # Main system loop
            self.is_running = True
            self._main_loop()
            
        except Exception as e:
            logger.error(f"Failed to start AUTOMOS AI: {e}")
            self.shutdown()
            raise
    
    def _initialize_components(self, mode: str, hardware: str):
        """Initialize system components based on mode and hardware"""
        
        # Initialize perception pipeline
        self.perception_pipeline.initialize(
            camera_count=6,  # 360° coverage
            radar_sensors=4,
            lidar_sensors=2,
            hardware=hardware
        )
        
        # Initialize reasoning engine
        self.reasoning_engine.initialize(
            llm_model="dualad_opendrivevla_fusion",
            planning_algorithm="lattice_idm_vla",
            hardware=hardware
        )
        
        # Initialize safety critic with <10ms response requirement
        self.safety_critic.initialize(
            response_time_ms=10,
            emergency_override=True,
            hardware=hardware
        )
        
        # Initialize world model
        self.world_model.initialize(
            hd_map_free=True,
            predictive_capability=True,
            hardware=hardware
        )
    
    def _main_loop(self):
        """Main system loop"""
        logger.info("Entering main system loop")
        
        while self.is_running:
            try:
                # Get sensor data
                sensor_data = self.perception_pipeline.process_sensors()
                
                # Update world model
                world_state = self.world_model.update(sensor_data)
                
                # Safety check (must complete in <10ms)
                safety_result = self.safety_critic.evaluate(world_state)
                
                if safety_result.emergency_stop_required:
                    logger.warning("Emergency stop triggered by safety critic")
                    self._emergency_stop()
                    continue
                
                # Generate driving plan
                driving_plan = self.reasoning_engine.generate_plan(
                    world_state=world_state,
                    safety_constraints=safety_result.constraints,
                    language_command="Navigate safely to destination"
                )
                
                # Execute plan
                self.system_coordinator.execute_plan(driving_plan)
                
            except KeyboardInterrupt:
                logger.info("Shutdown signal received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                continue
    
    def _emergency_stop(self):
        """Execute emergency stop procedure"""
        logger.critical("EMERGENCY STOP ACTIVATED")
        self.emergency_override = True
        self.system_coordinator.emergency_stop()
    
    def shutdown(self):
        """Shutdown the AUTOMOS AI system"""
        logger.info("Shutting down AUTOMOS AI...")
        self.is_running = False
        
        # Shutdown components
        if hasattr(self, 'system_coordinator'):
            self.system_coordinator.shutdown()
        if hasattr(self, 'perception_pipeline'):
            self.perception_pipeline.shutdown()
        if hasattr(self, 'reasoning_engine'):
            self.reasoning_engine.shutdown()
        
        logger.info("AUTOMOS AI shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AUTOMOS AI - Reasoning-Based Autonomous Driving")
    parser.add_argument("--mode", choices=["simulation", "deployment"], default="simulation",
                       help="Operation mode")
    parser.add_argument("--hardware", choices=["cpu", "gpu", "edge"], default="cpu",
                       help="Target hardware")
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    args = parser.parse_args()
    
    # Create and start AUTOMOS AI
    automos = AUTOMOS_AI(args.config)
    
    try:
        automos.start(mode=args.mode, hardware=args.hardware)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        automos.shutdown()

if __name__ == "__main__":
    main()
