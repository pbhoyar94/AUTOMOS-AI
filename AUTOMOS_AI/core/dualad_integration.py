"""
AUTOMOS AI DualAD Integration
Integrates DualAD dual-layer planning system
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)

class DualADIntegration:
    """Integration layer for DualAD framework"""
    
    def __init__(self):
        """Initialize DualAD integration"""
        logger.info("Initializing DualAD Integration...")
        
        # Path to DualAD source code
        self.dualad_path = Path(__file__).parent.parent.parent / "DualAD-main"
        
        # Components
        self.llm_interface = None
        self.planning_module = None
        self.simulation_interface = None
        
        # Configuration
        self.config = {
            'llm_model': 'GLM-4-Flash',  # Free model option
            'planning_algorithm': 'Lattice-IDM',
            'use_llm': True,
            'api_keys': {}
        }
        
        logger.info("DualAD Integration initialized")
    
    def initialize(self, llm_model: str, hardware: str):
        """
        Initialize DualAD components
        
        Args:
            llm_model: LLM model to use
            hardware: Target hardware platform
        """
        logger.info(f"Initializing DualAD with {llm_model} on {hardware}")
        
        try:
            # Add DualAD to Python path
            if self.dualad_path.exists():
                sys.path.insert(0, str(self.dualad_path))
                logger.info(f"Added DualAD path: {self.dualad_path}")
            else:
                logger.warning(f"DualAD path not found: {self.dualad_path}")
                self._create_mock_dualad()
                return
            
            # Initialize LLM interface
            self._initialize_llm_interface(llm_model)
            
            # Initialize planning module
            self._initialize_planning_module(hardware)
            
            # Load configuration
            self._load_configuration()
            
            logger.info("DualAD components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DualAD: {e}")
            self._create_mock_dualad()
    
    def _initialize_llm_interface(self, llm_model: str):
        """Initialize LLM interface for reasoning"""
        
        try:
            # Try to import DualAD LLM components
            from nuplan.planning.training.modeling.models.llm_wrapper import LLMWrapper
            
            self.llm_interface = LLMWrapper(
                model_name=llm_model,
                api_key=self.config.get('api_keys', {}).get('openai'),
                use_openai='gpt' in llm_model.lower()
            )
            
            logger.info(f"Initialized LLM interface with {llm_model}")
            
        except ImportError:
            logger.warning("DualAD LLM components not available, creating mock interface")
            self.llm_interface = MockLLMInterface(llm_model)
    
    def _initialize_planning_module(self, hardware: str):
        """Initialize planning module"""
        
        try:
            # Try to import DualAD planning components
            from nuplan.planning.training.modeling.models.lattice_idm_planner import LatticeIDMPlanner
            
            self.planning_module = LatticeIDMPlanner(
                device='cuda' if hardware == 'gpu' else 'cpu'
            )
            
            logger.info(f"Initialized Lattice-IDM planner for {hardware}")
            
        except ImportError:
            logger.warning("DualAD planning components not available, creating mock planner")
            self.planning_module = MockLatticeIDMPlanner(hardware)
    
    def _load_configuration(self):
        """Load DualAD configuration"""
        
        config_file = self.dualad_path / "LLM.yml"
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                self.config.update(config_data)
                logger.info("Loaded DualAD configuration")
                
            except Exception as e:
                logger.warning(f"Failed to load DualAD config: {e}")
    
    def generate_reasoning_plan(self, world_state: Dict, scenario_context: str) -> Dict:
        """
        Generate reasoning-based plan using DualAD
        
        Args:
            world_state: Current world state
            scenario_context: Text description of scenario
            
        Returns:
            Planning result with trajectory and reasoning
        """
        
        try:
            if self.config.get('use_llm', False) and self.llm_interface:
                # Use LLM-enhanced planning
                return self._llm_enhanced_planning(world_state, scenario_context)
            else:
                # Use rule-based planning
                return self._rule_based_planning(world_state)
                
        except Exception as e:
            logger.error(f"DualAD planning failed: {e}")
            return self._emergency_planning(world_state)
    
    def _llm_enhanced_planning(self, world_state: Dict, scenario_context: str) -> Dict:
        """LLM-enhanced planning"""
        
        # Generate LLM reasoning
        llm_reasoning = self.llm_interface.generate_reasoning(
            scenario_context=scenario_context,
            world_state=world_state
        )
        
        # Generate base trajectory
        base_trajectory = self.planning_module.generate_trajectory(world_state)
        
        # Enhance trajectory with LLM reasoning
        enhanced_trajectory = self._enhance_trajectory_with_reasoning(
            base_trajectory, llm_reasoning
        )
        
        return {
            'trajectory': enhanced_trajectory,
            'reasoning': llm_reasoning,
            'confidence': 0.85,
            'method': 'llm_enhanced'
        }
    
    def _rule_based_planning(self, world_state: Dict) -> Dict:
        """Rule-based planning"""
        
        trajectory = self.planning_module.generate_trajectory(world_state)
        
        return {
            'trajectory': trajectory,
            'reasoning': 'Rule-based Lattice-IDM planning',
            'confidence': 0.75,
            'method': 'rule_based'
        }
    
    def _enhance_trajectory_with_reasoning(self, trajectory: np.ndarray, reasoning: str) -> np.ndarray:
        """Enhance trajectory based on LLM reasoning"""
        
        # Simple enhancement: adjust velocity based on reasoning content
        enhanced_trajectory = trajectory.copy()
        
        # Parse reasoning for speed adjustments
        if 'slow' in reasoning.lower() or 'cautious' in reasoning.lower():
            # Reduce velocities by 20%
            enhanced_trajectory[:, 3] *= 0.8  # Assuming velocity is in column 3
        elif 'fast' in reasoning.lower() or 'efficient' in reasoning.lower():
            # Increase velocities by 10% if safe
            enhanced_trajectory[:, 3] *= 1.1
        
        return enhanced_trajectory
    
    def _emergency_planning(self, world_state: Dict) -> Dict:
        """Emergency planning fallback"""
        
        # Simple emergency trajectory
        emergency_trajectory = np.zeros((50, 4))  # 5 second horizon, 100ms steps
        
        # Initialize with current state
        current_speed = world_state.get('ego_vehicle_state', {}).get('speed', 0)
        emergency_trajectory[:, 3] = np.maximum(0, current_speed - 8.0 * np.linspace(0, 5, 50))  # Max braking
        
        return {
            'trajectory': emergency_trajectory,
            'reasoning': 'Emergency braking procedure',
            'confidence': 0.95,
            'method': 'emergency'
        }
    
    def _create_mock_dualad(self):
        """Create mock DualAD components when source is not available"""
        
        logger.info("Creating mock DualAD components")
        
        self.llm_interface = MockLLMInterface(self.config.get('llm_model', 'GLM-4-Flash'))
        self.planning_module = MockLatticeIDMPlanner('cpu')
        
        logger.info("Mock DualAD components created")
    
    def shutdown(self):
        """Shutdown DualAD integration"""
        logger.info("Shutting down DualAD integration...")
        
        if self.llm_interface:
            self.llm_interface.shutdown()
        if self.planning_module:
            self.planning_module.shutdown()
        
        logger.info("DualAD integration shutdown complete")

class MockLLMInterface:
    """Mock LLM interface for testing"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Created mock LLM interface for {model_name}")
    
    def generate_reasoning(self, scenario_context: str, world_state: Dict) -> str:
        """Generate mock reasoning"""
        
        # Simple rule-based reasoning
        if 'intersection' in scenario_context.lower():
            return "Approaching intersection, proceed with caution and check for cross traffic"
        elif 'highway' in scenario_context.lower():
            return "On highway, maintain safe following distance and lane discipline"
        elif 'emergency' in scenario_context.lower():
            return "Emergency situation detected, prepare to stop or yield"
        else:
            return "Normal driving conditions, proceed with standard safety protocols"
    
    def shutdown(self):
        """Shutdown mock LLM"""
        pass

class MockLatticeIDMPlanner:
    """Mock Lattice-IDM planner for testing"""
    
    def __init__(self, hardware: str):
        self.hardware = hardware
        logger.info(f"Created mock Lattice-IDM planner for {hardware}")
    
    def generate_trajectory(self, world_state: Dict) -> np.ndarray:
        """Generate mock trajectory"""
        
        # Simple constant velocity trajectory
        horizon = 5.0  # 5 seconds
        dt = 0.1  # 100ms steps
        steps = int(horizon / dt)
        
        trajectory = np.zeros((steps, 4))  # x, y, theta, velocity
        
        current_speed = world_state.get('ego_vehicle_state', {}).get('speed', 10.0)  # m/s
        
        for i in range(steps):
            t = i * dt
            trajectory[i, 0] = current_speed * t  # x position
            trajectory[i, 1] = 0  # y position (straight line)
            trajectory[i, 2] = 0  # heading (straight)
            trajectory[i, 3] = current_speed  # constant velocity
        
        return trajectory
    
    def shutdown(self):
        """Shutdown mock planner"""
        pass
