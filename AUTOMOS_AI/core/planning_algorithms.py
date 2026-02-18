"""
AUTOMOS AI Planning Algorithms
Implements Lattice-IDM and other planning algorithms
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

logger = logging.getLogger(__name__)

class PlanningMode(Enum):
    """Planning algorithm modes"""
    LATTICE_IDM = "lattice_idm"
    VLA_PLANNER = "vla_planner"
    EMERGENCY = "emergency"
    RULE_BASED = "rule_based"

@dataclass
class PlanningResult:
    """Result from planning algorithm"""
    trajectory: np.ndarray
    velocity_profile: np.ndarray
    steering_profile: np.ndarray
    confidence: float
    computation_time_ms: float
    algorithm_used: str

class LatticeIDMPlanner:
    """Lattice-IDM planning algorithm implementation"""
    
    def __init__(self):
        """Initialize Lattice-IDM planner"""
        logger.info("Initializing Lattice-IDM Planner...")
        
        # IDM parameters
        self.idm_params = {
            'desired_speed': 15.0,  # m/s (54 km/h)
            'time_headway': 1.5,     # seconds
            'min_spacing': 2.0,      # meters
            'max_acceleration': 2.0,  # m/s²
            'comfortable_deceleration': 1.5,  # m/s²
            'acceleration_exponent': 4
        }
        
        # Lattice parameters
        self.lattice_params = {
            'longitudinal_samples': 7,
            'lateral_samples': 7,
            'temporal_samples': 8,
            'longitudinal_spacing': 2.0,  # meters
            'lateral_spacing': 0.5,       # meters
            'time_horizon': 3.0,          # seconds
            'dt': 0.1                     # seconds
        }
        
        # Vehicle parameters
        self.vehicle_params = {
            'wheelbase': 2.5,      # meters
            'max_steering_angle': 0.6,  # radians
            'max_steering_rate': 1.0,   # rad/s
            'max_acceleration': 3.0,    # m/s²
            'max_deceleration': 8.0     # m/s²
        }
        
        logger.info("Lattice-IDM Planner initialized")
    
    def initialize(self, hardware: str):
        """Initialize planner for specific hardware"""
        self.device = 'cuda' if hardware == 'gpu' and torch.cuda.is_available() else 'cpu'
        logger.info(f"Lattice-IDM planner initialized for {hardware}")
    
    def generate_plan(self, world_state: Dict, safety_constraints: List[str]) -> PlanningResult:
        """
        Generate driving plan using Lattice-IDM algorithm
        
        Args:
            world_state: Current world state
            safety_constraints: Safety constraints to respect
            
        Returns:
            PlanningResult: Generated plan
        """
        import time
        start_time = time.time()
        
        try:
            # Extract relevant information from world state
            ego_state = world_state.get('ego_vehicle_state', {})
            surrounding_objects = world_state.get('surrounding_objects', [])
            road_geometry = world_state.get('road_geometry', {})
            
            # Generate lattice of candidate trajectories
            candidate_trajectories = self._generate_lattice_trajectories(ego_state, road_geometry)
            
            # Evaluate trajectories using IDM model
            evaluated_trajectories = self._evaluate_trajectories_with_idm(
                candidate_trajectories, ego_state, surrounding_objects
            )
            
            # Select best trajectory
            best_trajectory = self._select_best_trajectory(evaluated_trajectories)
            
            # Generate velocity and steering profiles
            velocity_profile = self._generate_velocity_profile(best_trajectory, ego_state, surrounding_objects)
            steering_profile = self._generate_steering_profile(best_trajectory)
            
            # Calculate computation time
            computation_time = (time.time() - start_time) * 1000
            
            result = PlanningResult(
                trajectory=best_trajectory,
                velocity_profile=velocity_profile,
                steering_profile=steering_profile,
                confidence=0.85,
                computation_time_ms=computation_time,
                algorithm_used="Lattice-IDM"
            )
            
            logger.debug(f"Generated Lattice-IDM plan in {computation_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Lattice-IDM planning failed: {e}")
            return self._emergency_plan(world_state)
    
    def _generate_lattice_trajectories(self, ego_state: Dict, road_geometry: Dict) -> List[np.ndarray]:
        """Generate lattice of candidate trajectories"""
        
        trajectories = []
        
        # Current state
        current_x = ego_state.get('x', 0.0)
        current_y = ego_state.get('y', 0.0)
        current_theta = ego_state.get('heading', 0.0)
        current_speed = ego_state.get('speed', 10.0)
        
        # Generate longitudinal samples
        for i in range(self.lattice_params['longitudinal_samples']):
            longitudinal_offset = (i - self.lattice_params['longitudinal_samples'] // 2) * self.lattice_params['longitudinal_spacing']
            
            # Generate lateral samples
            for j in range(self.lattice_params['lateral_samples']):
                lateral_offset = (j - self.lattice_params['lateral_samples'] // 2) * self.lattice_params['lateral_spacing']
                
                # Generate trajectory
                trajectory = self._generate_single_trajectory(
                    current_x, current_y, current_theta, current_speed,
                    longitudinal_offset, lateral_offset
                )
                
                trajectories.append(trajectory)
        
        return trajectories
    
    def _generate_single_trajectory(self, x0: float, y0: float, theta0: float, v0: float,
                                   longitudinal_offset: float, lateral_offset: float) -> np.ndarray:
        """Generate a single trajectory"""
        
        steps = int(self.lattice_params['time_horizon'] / self.lattice_params['dt'])
        trajectory = np.zeros((steps, 3))  # x, y, theta
        
        # Simple trajectory generation
        for i in range(steps):
            t = i * self.lattice_params['dt']
            
            # Longitudinal motion
            if longitudinal_offset > 0:
                # Accelerating trajectory
                v = v0 + 2.0 * t
            elif longitudinal_offset < 0:
                # Decelerating trajectory
                v = max(0, v0 - 2.0 * t)
            else:
                # Constant velocity
                v = v0
            
            # Lateral motion
            if lateral_offset != 0:
                # Simple lateral maneuver
                lateral_progress = min(1.0, t / 2.0)  # Complete maneuver in 2 seconds
                y = y0 + lateral_offset * lateral_progress
                theta = theta0 + lateral_offset * 0.1 * np.sin(np.pi * lateral_progress)
            else:
                y = y0
                theta = theta0
            
            # Update position
            x = x0 + v * t * np.cos(theta)
            
            trajectory[i] = [x, y, theta]
        
        return trajectory
    
    def _evaluate_trajectories_with_idm(self, trajectories: List[np.ndarray], 
                                       ego_state: Dict, surrounding_objects: List[Dict]) -> List[Tuple[np.ndarray, float]]:
        """Evaluate trajectories using Intelligent Driver Model"""
        
        evaluated = []
        
        for trajectory in trajectories:
            cost = self._calculate_trajectory_cost(trajectory, ego_state, surrounding_objects)
            evaluated.append((trajectory, cost))
        
        return evaluated
    
    def _calculate_trajectory_cost(self, trajectory: np.ndarray, ego_state: Dict, 
                                  surrounding_objects: List[Dict]) -> float:
        """Calculate cost for a trajectory"""
        
        cost = 0.0
        
        # Progress cost (prefer forward progress)
        final_x = trajectory[-1, 0]
        initial_x = trajectory[0, 0]
        progress = final_x - initial_x
        cost -= progress * 0.1  # Negative cost for progress
        
        # Comfort cost (penalize high accelerations)
        if len(trajectory) > 1:
            velocities = np.diff(trajectory[:, 0]) / self.lattice_params['dt']
            accelerations = np.diff(velocities) / self.lattice_params['dt']
            cost += np.sum(np.abs(accelerations)) * 0.1
        
        # Collision cost
        collision_cost = self._calculate_collision_cost(trajectory, surrounding_objects)
        cost += collision_cost
        
        # Lane keeping cost
        lane_cost = self._calculate_lane_keeping_cost(trajectory)
        cost += lane_cost
        
        return cost
    
    def _calculate_collision_cost(self, trajectory: np.ndarray, surrounding_objects: List[Dict]) -> float:
        """Calculate collision risk cost"""
        
        collision_cost = 0.0
        
        for obj in surrounding_objects:
            obj_x = obj.get('x', 0)
            obj_y = obj.get('y', 0)
            obj_vx = obj.get('vx', 0)
            obj_vy = obj.get('vy', 0)
            obj_radius = obj.get('radius', 2.0)
            
            for i, point in enumerate(trajectory):
                t = i * self.lattice_params['dt']
                
                # Predict object position
                pred_obj_x = obj_x + obj_vx * t
                pred_obj_y = obj_y + obj_vy * t
                
                # Calculate distance
                distance = np.sqrt((point[0] - pred_obj_x)**2 + (point[1] - pred_obj_y)**2)
                
                # Add cost if too close
                if distance < obj_radius + 1.0:  # 1m safety margin
                    collision_cost += 100.0 * (1.0 - distance / (obj_radius + 1.0))
        
        return collision_cost
    
    def _calculate_lane_keeping_cost(self, trajectory: np.ndarray) -> float:
        """Calculate lane keeping cost"""
        
        # Simple lane keeping: penalize lateral deviation
        lateral_deviations = np.abs(trajectory[:, 1])
        return np.sum(lateral_deviations) * 0.01
    
    def _select_best_trajectory(self, evaluated_trajectories: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Select best trajectory from evaluated candidates"""
        
        if not evaluated_trajectories:
            return self._generate_default_trajectory()
        
        # Sort by cost (lower is better)
        evaluated_trajectories.sort(key=lambda x: x[1])
        
        # Return best trajectory
        return evaluated_trajectories[0][0]
    
    def _generate_default_trajectory(self) -> np.ndarray:
        """Generate default straight trajectory"""
        
        steps = int(self.lattice_params['time_horizon'] / self.lattice_params['dt'])
        trajectory = np.zeros((steps, 3))
        
        for i in range(steps):
            t = i * self.lattice_params['dt']
            trajectory[i] = [10.0 * t, 0.0, 0.0]  # Straight line at 10 m/s
        
        return trajectory
    
    def _generate_velocity_profile(self, trajectory: np.ndarray, ego_state: Dict, 
                                  surrounding_objects: List[Dict]) -> np.ndarray:
        """Generate velocity profile for trajectory"""
        
        steps = len(trajectory)
        velocity_profile = np.zeros(steps)
        
        current_speed = ego_state.get('speed', 10.0)
        desired_speed = self.idm_params['desired_speed']
        
        for i in range(steps):
            # Calculate IDM speed
            if i == 0:
                velocity_profile[i] = current_speed
            else:
                # Simple IDM implementation
                distance_to_front = self._get_distance_to_front_vehicle(trajectory[i], surrounding_objects)
                
                if distance_to_front < 10.0:  # Close to vehicle ahead
                    # Slow down
                    velocity_profile[i] = min(current_speed, distance_to_front / 2.0)
                else:
                    # Accelerate to desired speed
                    velocity_profile[i] = min(desired_speed, current_speed + self.idm_params['max_acceleration'] * self.lattice_params['dt'])
                
                current_speed = velocity_profile[i]
        
        return velocity_profile
    
    def _get_distance_to_front_vehicle(self, position: np.ndarray, surrounding_objects: List[Dict]) -> float:
        """Get distance to front vehicle"""
        
        min_distance = float('inf')
        
        for obj in surrounding_objects:
            obj_x = obj.get('x', 0)
            obj_y = obj.get('y', 0)
            
            # Check if object is in front
            if obj_x > position[0]:
                distance = np.sqrt((obj_x - position[0])**2 + (obj_y - position[1])**2)
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 100.0
    
    def _generate_steering_profile(self, trajectory: np.ndarray) -> np.ndarray:
        """Generate steering profile for trajectory"""
        
        steps = len(trajectory)
        steering_profile = np.zeros(steps)
        
        for i in range(1, steps):
            # Calculate heading change
            delta_theta = trajectory[i, 2] - trajectory[i-1, 2]
            dt = self.lattice_params['dt']
            
            # Convert to steering angle (simplified)
            steering_profile[i] = delta_theta / dt * self.vehicle_params['wheelbase']
            
            # Limit steering angle
            steering_profile[i] = np.clip(steering_profile[i], 
                                          -self.vehicle_params['max_steering_angle'],
                                          self.vehicle_params['max_steering_angle'])
        
        return steering_profile
    
    def _emergency_plan(self, world_state: Dict) -> PlanningResult:
        """Generate emergency plan"""
        
        steps = int(3.0 / 0.1)  # 3 second horizon
        trajectory = np.zeros((steps, 3))
        velocity_profile = np.zeros(steps)
        steering_profile = np.zeros(steps)
        
        # Emergency braking
        current_speed = world_state.get('ego_vehicle_state', {}).get('speed', 10.0)
        for i in range(steps):
            velocity_profile[i] = max(0, current_speed - 8.0 * i * 0.1)  # Max deceleration
            trajectory[i, 0] = np.sum(velocity_profile[:i+1]) * 0.1
        
        return PlanningResult(
            trajectory=trajectory,
            velocity_profile=velocity_profile,
            steering_profile=steering_profile,
            confidence=0.95,
            computation_time_ms=1.0,
            algorithm_used="Emergency"
        )
    
    def shutdown(self):
        """Shutdown planner"""
        logger.info("Lattice-IDM planner shutdown")

class VLAPlanner:
    """Vision-Language-Action planner"""
    
    def __init__(self):
        """Initialize VLA planner"""
        logger.info("Initializing VLA Planner...")
        
    def initialize(self, hardware: str):
        """Initialize VLA planner"""
        self.device = 'cuda' if hardware == 'gpu' and torch.cuda.is_available() else 'cpu'
        logger.info(f"VLA planner initialized for {hardware}")
    
    def generate_plan(self, world_state: Dict, language_command: str, 
                     safety_constraints: List[str]) -> PlanningResult:
        """Generate plan using VLA approach"""
        
        # This would integrate with the VLA model
        # For now, return a simple plan
        steps = int(5.0 / 0.1)
        trajectory = np.zeros((steps, 3))
        velocity_profile = np.ones(steps) * 12.0  # 12 m/s
        steering_profile = np.zeros(steps)
        
        for i in range(steps):
            t = i * 0.1
            trajectory[i, 0] = 12.0 * t
        
        return PlanningResult(
            trajectory=trajectory,
            velocity_profile=velocity_profile,
            steering_profile=steering_profile,
            confidence=0.8,
            computation_time_ms=50.0,
            algorithm_used="VLA"
        )
    
    def shutdown(self):
        """Shutdown VLA planner"""
        logger.info("VLA planner shutdown")
