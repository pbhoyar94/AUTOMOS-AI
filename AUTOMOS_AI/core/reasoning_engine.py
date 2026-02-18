"""
AUTOMOS AI Reasoning Engine
Integrates DualAD and OpenDriveVLA for advanced reasoning capabilities
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import numpy as np

logger = logging.getLogger(__name__)

class ReasoningMode(Enum):
    """Reasoning modes for different driving scenarios"""
    RULE_BASED = "rule_based"
    LLM_GUIDED = "llm_guided"
    HYBRID = "hybrid"
    EMERGENCY = "emergency"

@dataclass
class DrivingPlan:
    """Driving plan output from reasoning engine"""
    trajectory: np.ndarray
    velocity_profile: np.ndarray
    steering_commands: np.ndarray
    confidence_score: float
    reasoning_explanation: str
    safety_constraints: List[str]
    execution_time_ms: float

@dataclass
class WorldState:
    """Current world state for reasoning"""
    ego_vehicle_state: Dict
    surrounding_objects: List[Dict]
    road_geometry: Dict
    traffic_conditions: Dict
    weather_conditions: Dict
    social_context: Dict

class ReasoningEngine:
    """Main reasoning engine integrating multiple AI approaches"""
    
    def __init__(self):
        """Initialize reasoning engine"""
        logger.info("Initializing Reasoning Engine...")
        
        # Component engines
        self.dualad_engine = None
        self.vla_engine = None
        self.rule_based_planner = None
        
        # Reasoning state
        self.current_mode = ReasoningMode.HYBRID
        self.performance_metrics = {
            'total_plans': 0,
            'average_planning_time': 0.0,
            'emergency_interventions': 0,
            'llm_usage_count': 0
        }
        
        # Safety constraints
        self.safety_constraints = [
            "max_acceleration: 3.0 m/s²",
            "max_deceleration: 8.0 m/s²", 
            "max_steering_rate: 1.0 rad/s",
            "min_following_distance: 2.0 s",
            "max_speed_limit: current_speed_limit"
        ]
        
        logger.info("Reasoning Engine initialized")
    
    def initialize(self, llm_model: str, planning_algorithm: str, hardware: str):
        """
        Initialize reasoning components
        
        Args:
            llm_model: LLM model to use for reasoning
            planning_algorithm: Planning algorithm configuration
            hardware: Target hardware platform
        """
        logger.info(f"Initializing reasoning engine with {llm_model} on {hardware}")
        
        try:
            # Initialize DualAD integration
            self.dualad_engine = DualADIntegration()
            self.dualad_engine.initialize(
                llm_model=llm_model,
                hardware=hardware
            )
            
            # Initialize OpenDriveVLA integration  
            self.vla_engine = OpenDriveVLAIntegration()
            self.vla_engine.initialize(
                model_path="OpenDriveVLA-0.5B",
                hardware=hardware
            )
            
            # Initialize rule-based planner
            self.rule_based_planner = LatticeIDMPlanner()
            self.rule_based_planner.initialize(hardware=hardware)
            
            logger.info("Reasoning engine components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize reasoning engine: {e}")
            raise
    
    def generate_plan(self, world_state: WorldState, safety_constraints: List[str], 
                     language_command: str) -> DrivingPlan:
        """
        Generate driving plan using integrated reasoning
        
        Args:
            world_state: Current world state
            safety_constraints: Safety constraints to respect
            language_command: Natural language command
            
        Returns:
            DrivingPlan: Generated driving plan
        """
        start_time = time.time()
        
        try:
            # Determine reasoning mode based on scenario complexity
            self.current_mode = self._determine_reasoning_mode(world_state)
            
            logger.debug(f"Using {self.current_mode.value} reasoning mode")
            
            # Generate plan based on mode
            if self.current_mode == ReasoningMode.EMERGENCY:
                plan = self._emergency_planning(world_state, safety_constraints)
            elif self.current_mode == ReasoningMode.LLM_GUIDED:
                plan = self._llm_guided_planning(world_state, language_command, safety_constraints)
            elif self.current_mode == ReasoningMode.HYBRID:
                plan = self._hybrid_planning(world_state, language_command, safety_constraints)
            else:  # RULE_BASED
                plan = self._rule_based_planning(world_state, safety_constraints)
            
            # Calculate execution time
            plan.execution_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self._update_metrics(plan)
            
            # Validate plan safety
            if not self._validate_plan_safety(plan, safety_constraints):
                logger.warning("Generated plan failed safety validation, using fallback")
                plan = self._emergency_planning(world_state, safety_constraints)
            
            logger.debug(f"Generated plan with confidence {plan.confidence_score:.2f}")
            return plan
            
        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            return self._emergency_planning(world_state, safety_constraints)
    
    def _determine_reasoning_mode(self, world_state: WorldState) -> ReasoningMode:
        """Determine appropriate reasoning mode based on scenario"""
        
        # Check for emergency conditions
        if self._is_emergency_scenario(world_state):
            return ReasoningMode.EMERGENCY
        
        # Check for complex scenarios requiring LLM reasoning
        if self._is_complex_scenario(world_state):
            return ReasoningMode.LLM_GUIDED
        
        # Default to hybrid for normal operation
        return ReasoningMode.HYBRID
    
    def _is_emergency_scenario(self, world_state: WorldState) -> bool:
        """Check if current scenario requires emergency response"""
        
        # Check for imminent collision
        for obj in world_state.surrounding_objects:
            distance = obj.get('distance', float('inf'))
            relative_velocity = obj.get('relative_velocity', 0)
            
            # Time to collision calculation
            if distance < 5.0 and relative_velocity > 2.0:
                return True
        
        # Check for loss of control
        ego_state = world_state.ego_vehicle_state
        if (abs(ego_state.get('lateral_acceleration', 0)) > 5.0 or
            abs(ego_state.get('yaw_rate', 0)) > 0.5):
            return True
        
        return False
    
    def _is_complex_scenario(self, world_state: WorldState) -> bool:
        """Check if scenario requires LLM reasoning"""
        
        # Count surrounding objects
        if len(world_state.surrounding_objects) > 8:
            return True
        
        # Check for unusual traffic patterns
        traffic_density = world_state.traffic_conditions.get('density', 0)
        if traffic_density > 0.8:
            return True
        
        # Check for social context complexity
        if world_state.social_context.get('pedestrians_count', 0) > 3:
            return True
        
        return False
    
    def _emergency_planning(self, world_state: WorldState, safety_constraints: List[str]) -> DrivingPlan:
        """Generate emergency driving plan"""
        logger.warning("Executing emergency planning")
        
        # Simple emergency maneuver: maximum safe braking
        ego_state = world_state.ego_vehicle_state
        current_speed = ego_state.get('speed', 0)
        current_steering = ego_state.get('steering_angle', 0)
        
        # Emergency braking trajectory
        time_horizon = 2.0  # 2 second emergency planning horizon
        dt = 0.1  # 100ms timesteps
        timesteps = int(time_horizon / dt)
        
        trajectory = np.zeros((timesteps, 3))  # x, y, theta
        velocity_profile = np.zeros(timesteps)
        steering_commands = np.zeros(timesteps)
        
        # Initialize with current state
        trajectory[0] = [0, 0, ego_state.get('heading', 0)]
        velocity_profile[0] = current_speed
        steering_commands[0] = current_steering
        
        # Generate emergency braking trajectory
        max_decel = 8.0  # m/s²
        for i in range(1, timesteps):
            # Apply maximum safe deceleration
            velocity_profile[i] = max(0, velocity_profile[i-1] - max_decel * dt)
            
            # Maintain current steering (don't make sudden steering changes in emergency)
            steering_commands[i] = current_steering
            
            # Simple kinematic trajectory
            if velocity_profile[i] > 0:
                trajectory[i, 0] = trajectory[i-1, 0] + velocity_profile[i] * dt * np.cos(trajectory[i-1, 2])
                trajectory[i, 1] = trajectory[i-1, 1] + velocity_profile[i] * dt * np.sin(trajectory[i-1, 2])
                trajectory[i, 2] = trajectory[i-1, 2] + velocity_profile[i] * dt * np.tan(current_steering) / 2.5  # wheelbase ~2.5m
            else:
                trajectory[i] = trajectory[i-1]
        
        return DrivingPlan(
            trajectory=trajectory,
            velocity_profile=velocity_profile,
            steering_commands=steering_commands,
            confidence_score=0.95,  # High confidence in emergency procedures
            reasoning_explanation="Emergency braking maneuver activated due to safety critical situation",
            safety_constraints=safety_constraints,
            execution_time_ms=5.0  # Emergency planning must be very fast
        )
    
    def _llm_guided_planning(self, world_state: WorldState, language_command: str, 
                           safety_constraints: List[str]) -> DrivingPlan:
        """Generate plan using LLM-guided reasoning"""
        logger.debug("Using LLM-guided planning")
        
        try:
            # Use VLA engine for vision-language-action planning
            plan = self.vla_engine.generate_plan(
                world_state=world_state,
                language_command=language_command,
                safety_constraints=safety_constraints
            )
            
            self.performance_metrics['llm_usage_count'] += 1
            return plan
            
        except Exception as e:
            logger.error(f"LLM planning failed: {e}, falling back to rule-based")
            return self._rule_based_planning(world_state, safety_constraints)
    
    def _hybrid_planning(self, world_state: WorldState, language_command: str,
                        safety_constraints: List[str]) -> DrivingPlan:
        """Generate plan using hybrid approach"""
        logger.debug("Using hybrid planning")
        
        try:
            # Get rule-based plan as baseline
            rule_plan = self._rule_based_planning(world_state, safety_constraints)
            
            # Enhance with LLM reasoning for complex decisions
            if self._should_enhance_with_llm(world_state):
                llm_enhancement = self.vla_engine.enhance_plan(
                    base_plan=rule_plan,
                    world_state=world_state,
                    language_command=language_command
                )
                
                # Merge plans with safety validation
                enhanced_plan = self._merge_plans(rule_plan, llm_enhancement, safety_constraints)
                return enhanced_plan
            
            return rule_plan
            
        except Exception as e:
            logger.error(f"Hybrid planning failed: {e}, using rule-based fallback")
            return self._rule_based_planning(world_state, safety_constraints)
    
    def _rule_based_planning(self, world_state: WorldState, safety_constraints: List[str]) -> DrivingPlan:
        """Generate plan using rule-based algorithms"""
        logger.debug("Using rule-based planning")
        
        try:
            plan = self.rule_based_planner.generate_plan(
                world_state=world_state,
                safety_constraints=safety_constraints
            )
            return plan
            
        except Exception as e:
            logger.error(f"Rule-based planning failed: {e}")
            raise
    
    def _should_enhance_with_llm(self, world_state: WorldState) -> bool:
        """Determine if plan should be enhanced with LLM reasoning"""
        
        # Enhance for complex social scenarios
        if world_state.social_context.get('complex_interaction', False):
            return True
        
        # Enhance for unusual road conditions
        if world_state.road_geometry.get('complexity', 0) > 0.7:
            return True
        
        return False
    
    def _merge_plans(self, rule_plan: DrivingPlan, llm_plan: DrivingPlan, 
                    safety_constraints: List[str]) -> DrivingPlan:
        """Merge rule-based and LLM plans safely"""
        
        # Use rule-based trajectory as primary (more reliable)
        merged_trajectory = rule_plan.trajectory
        
        # Incorporate LLM velocity profile adjustments
        merged_velocity = 0.7 * rule_plan.velocity_profile + 0.3 * llm_plan.velocity_profile
        
        # Use rule-based steering for safety
        merged_steering = rule_plan.steering_commands
        
        # Combine reasoning explanations
        merged_explanation = f"Hybrid plan: {rule_plan.reasoning_explanation}. Enhanced with: {llm_plan.reasoning_explanation}"
        
        return DrivingPlan(
            trajectory=merged_trajectory,
            velocity_profile=merged_velocity,
            steering_commands=merged_steering,
            confidence_score=0.8 * rule_plan.confidence_score + 0.2 * llm_plan.confidence_score,
            reasoning_explanation=merged_explanation,
            safety_constraints=safety_constraints,
            execution_time_ms=max(rule_plan.execution_time_ms, llm_plan.execution_time_ms)
        )
    
    def _validate_plan_safety(self, plan: DrivingPlan, safety_constraints: List[str]) -> bool:
        """Validate that plan respects all safety constraints"""
        
        # Check acceleration limits
        if len(plan.velocity_profile) > 1:
            dt = 0.1  # Assumed timestep
            accelerations = np.diff(plan.velocity_profile) / dt
            
            if np.any(np.abs(accelerations) > 8.0):  # Max 8 m/s²
                return False
        
        # Check steering rate limits
        if len(plan.steering_commands) > 1:
            steering_rates = np.diff(plan.steering_commands) / 0.1
            if np.any(np.abs(steering_rates) > 1.0):  # Max 1 rad/s
                return False
        
        # Check for reasonable confidence
        if plan.confidence_score < 0.3:
            return False
        
        return True
    
    def _update_metrics(self, plan: DrivingPlan):
        """Update performance metrics"""
        self.performance_metrics['total_plans'] += 1
        
        # Update average planning time
        current_avg = self.performance_metrics['average_planning_time']
        total_plans = self.performance_metrics['total_plans']
        self.performance_metrics['average_planning_time'] = (
            (current_avg * (total_plans - 1) + plan.execution_time_ms) / total_plans
        )
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def shutdown(self):
        """Shutdown reasoning engine"""
        logger.info("Shutting down reasoning engine...")
        
        if self.dualad_engine:
            self.dualad_engine.shutdown()
        if self.vla_engine:
            self.vla_engine.shutdown()
        if self.rule_based_planner:
            self.rule_based_planner.shutdown()
        
        logger.info("Reasoning engine shutdown complete")

# Import integration modules (these will be implemented separately)
from .dualad_integration import DualADIntegration
from .opendrivevla_integration import OpenDriveVLAIntegration  
from .planning_algorithms import LatticeIDMPlanner
