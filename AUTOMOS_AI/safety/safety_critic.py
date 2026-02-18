"""
AUTOMOS AI Safety Critic System
<10ms response time safety evaluation and emergency override
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import numpy as np

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety risk levels"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SafetyAction(Enum):
    """Safety actions"""
    CONTINUE = "continue"
    REDUCE_SPEED = "reduce_speed"
    INCREASE_FOLLOWING_DISTANCE = "increase_following_distance"
    PREPARE_TO_STOP = "prepare_to_stop"
    EMERGENCY_STOP = "emergency_stop"
    IMMEDIATE_BRAKE = "immediate_brake"

@dataclass
class SafetyAssessment:
    """Safety assessment result"""
    safety_level: SafetyLevel
    risk_score: float  # 0.0 (safe) to 1.0 (critical)
    recommended_action: SafetyAction
    constraints: List[str]
    reasoning: str
    response_time_ms: float
    emergency_stop_required: bool

@dataclass
class SafetyConstraint:
    """Safety constraint"""
    name: str
    value: float
    unit: str
    critical: bool
    description: str

class SafetyCritic:
    """Safety critic system with <10ms response requirement"""
    
    def __init__(self):
        """Initialize safety critic"""
        logger.info("Initializing Safety Critic System...")
        
        # Performance requirements
        self.target_response_time_ms = 10.0
        self.max_response_time_ms = 15.0
        
        # Safety thresholds
        self.safety_thresholds = {
            'max_acceleration': 3.0,  # m/s²
            'max_deceleration': 8.0,  # m/s²
            'max_lateral_acceleration': 4.0,  # m/s²
            'max_steering_rate': 1.0,  # rad/s
            'min_following_distance': 2.0,  # seconds
            'min_ttc': 3.0,  # Time to collision (seconds)
            'max_speed_deviation': 5.0,  # m/s from speed limit
            'max_lane_deviation': 0.5,  # meters
            'max_yaw_rate': 0.5  # rad/s
        }
        
        # Safety constraints
        self.active_constraints = []
        self.emergency_constraints = []
        
        # Assessment history
        self.assessment_history = []
        self.max_history_size = 100
        
        # Performance metrics
        self.metrics = {
            'total_assessments': 0,
            'average_response_time': 0.0,
            'emergency_activations': 0,
            'constraint_violations': 0,
            'max_response_time': 0.0
        }
        
        # Thread safety
        self.assessment_lock = threading.Lock()
        
        logger.info("Safety Critic System initialized")
    
    def initialize(self, response_time_ms: float, emergency_override: bool, hardware: str):
        """
        Initialize safety critic with configuration
        
        Args:
            response_time_ms: Target response time in milliseconds
            emergency_override: Enable emergency override capability
            hardware: Target hardware platform
        """
        logger.info(f"Initializing Safety Critic with {response_time_ms}ms target response on {hardware}")
        
        try:
            # Update configuration
            self.target_response_time_ms = response_time_ms
            self.emergency_override_enabled = emergency_override
            self.hardware = hardware
            
            # Initialize emergency override system
            if emergency_override:
                self._initialize_emergency_system()
            
            # Set up safety constraints
            self._setup_safety_constraints()
            
            # Optimize for target hardware
            self._optimize_for_hardware(hardware)
            
            logger.info("Safety Critic initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Safety Critic: {e}")
            raise
    
    def _initialize_emergency_system(self):
        """Initialize emergency override system"""
        
        self.emergency_system = {
            'active': False,
            'activation_time': None,
            'reason': None,
            'override_level': 0
        }
        
        logger.info("Emergency override system initialized")
    
    def _setup_safety_constraints(self):
        """Set up safety constraints"""
        
        self.active_constraints = [
            SafetyConstraint(
                name="max_acceleration",
                value=self.safety_thresholds['max_acceleration'],
                unit="m/s²",
                critical=False,
                description="Maximum longitudinal acceleration"
            ),
            SafetyConstraint(
                name="max_deceleration",
                value=self.safety_thresholds['max_deceleration'],
                unit="m/s²",
                critical=True,
                description="Maximum longitudinal deceleration"
            ),
            SafetyConstraint(
                name="max_lateral_acceleration",
                value=self.safety_thresholds['max_lateral_acceleration'],
                unit="m/s²",
                critical=True,
                description="Maximum lateral acceleration"
            ),
            SafetyConstraint(
                name="min_following_distance",
                value=self.safety_thresholds['min_following_distance'],
                unit="seconds",
                critical=True,
                description="Minimum time headway"
            ),
            SafetyConstraint(
                name="min_ttc",
                value=self.safety_thresholds['min_ttc'],
                unit="seconds",
                critical=True,
                description="Minimum time to collision"
            )
        ]
        
        logger.info(f"Set up {len(self.active_constraints)} safety constraints")
    
    def _optimize_for_hardware(self, hardware: str):
        """Optimize safety critic for specific hardware"""
        
        if hardware == 'edge':
            # Reduce computation for edge devices
            self.safety_thresholds['max_assessment_complexity'] = 'low'
            self.max_history_size = 50
        elif hardware == 'gpu':
            # Enable advanced safety analysis
            self.safety_thresholds['max_assessment_complexity'] = 'high'
            self.enable_advanced_analysis = True
        else:
            # Standard CPU configuration
            self.safety_thresholds['max_assessment_complexity'] = 'medium'
        
        logger.info(f"Optimized for {hardware} hardware")
    
    def evaluate(self, world_state: Dict) -> SafetyAssessment:
        """
        Evaluate safety of current world state
        
        Args:
            world_state: Current world state
            
        Returns:
            SafetyAssessment: Safety assessment result
        """
        start_time = time.time()
        
        try:
            with self.assessment_lock:
                # Extract relevant information
                ego_state = world_state.get('ego_vehicle_state', {})
                surrounding_objects = world_state.get('surrounding_objects', [])
                road_conditions = world_state.get('road_conditions', {})
                weather_conditions = world_state.get('weather_conditions', {})
                
                # Perform safety assessment
                risk_score = self._calculate_risk_score(ego_state, surrounding_objects, road_conditions)
                safety_level = self._determine_safety_level(risk_score)
                recommended_action = self._determine_safety_action(safety_level, risk_score)
                constraints = self._check_constraints(ego_state, surrounding_objects)
                reasoning = self._generate_safety_reasoning(safety_level, risk_score, constraints)
                
                # Check for emergency conditions
                emergency_stop_required = self._check_emergency_conditions(ego_state, surrounding_objects)
                
                # Calculate response time
                response_time = (time.time() - start_time) * 1000
                
                # Create assessment
                assessment = SafetyAssessment(
                    safety_level=safety_level,
                    risk_score=risk_score,
                    recommended_action=recommended_action,
                    constraints=constraints,
                    reasoning=reasoning,
                    response_time_ms=response_time,
                    emergency_stop_required=emergency_stop_required
                )
                
                # Update metrics
                self._update_metrics(assessment)
                
                # Store in history
                self._store_assessment(assessment)
                
                # Check response time requirement
                if response_time > self.max_response_time_ms:
                    logger.warning(f"Safety assessment exceeded max response time: {response_time:.2f}ms")
                
                logger.debug(f"Safety assessment completed in {response_time:.2f}ms: {safety_level.value}")
                return assessment
                
        except Exception as e:
            logger.error(f"Safety assessment failed: {e}")
            # Return emergency assessment on failure
            return self._emergency_assessment()
    
    def _calculate_risk_score(self, ego_state: Dict, surrounding_objects: List[Dict], 
                             road_conditions: Dict) -> float:
        """Calculate overall risk score (0.0 to 1.0)"""
        
        risk_score = 0.0
        
        # Collision risk (40% weight)
        collision_risk = self._calculate_collision_risk(ego_state, surrounding_objects)
        risk_score += 0.4 * collision_risk
        
        # Control loss risk (25% weight)
        control_risk = self._calculate_control_loss_risk(ego_state, road_conditions)
        risk_score += 0.25 * control_risk
        
        # Road departure risk (20% weight)
        departure_risk = self._calculate_road_departure_risk(ego_state, road_conditions)
        risk_score += 0.2 * departure_risk
        
        # Speed risk (15% weight)
        speed_risk = self._calculate_speed_risk(ego_state, road_conditions)
        risk_score += 0.15 * speed_risk
        
        return min(1.0, risk_score)
    
    def _calculate_collision_risk(self, ego_state: Dict, surrounding_objects: List[Dict]) -> float:
        """Calculate collision risk"""
        
        max_risk = 0.0
        
        for obj in surrounding_objects:
            distance = obj.get('distance', float('inf'))
            relative_velocity = obj.get('relative_velocity', 0)
            time_to_collision = obj.get('ttc', float('inf'))
            
            # Calculate risk based on TTC and distance
            if time_to_collision < 10.0:  # Within 10 seconds
                if time_to_collision < 3.0:
                    risk = 1.0  # Critical
                elif time_to_collision < 5.0:
                    risk = 0.7  # High
                elif time_to_collision < 8.0:
                    risk = 0.4  # Medium
                else:
                    risk = 0.2  # Low
                
                max_risk = max(max_risk, risk)
        
        return max_risk
    
    def _calculate_control_loss_risk(self, ego_state: Dict, road_conditions: Dict) -> float:
        """Calculate control loss risk"""
        
        risk = 0.0
        
        # Check for high speeds in poor conditions
        speed = ego_state.get('speed', 0)
        road_friction = road_conditions.get('friction_coefficient', 1.0)
        
        if road_friction < 0.5 and speed > 15:  # Low friction, high speed
            risk = 0.8
        elif road_friction < 0.7 and speed > 20:
            risk = 0.5
        elif speed > 30:  # High speed regardless
            risk = 0.3
        
        # Check for extreme maneuvers
        lateral_acceleration = ego_state.get('lateral_acceleration', 0)
        if abs(lateral_acceleration) > 4.0:
            risk = max(risk, 0.7)
        
        return min(1.0, risk)
    
    def _calculate_road_departure_risk(self, ego_state: Dict, road_conditions: Dict) -> float:
        """Calculate road departure risk"""
        
        risk = 0.0
        
        # Check lane position
        lane_offset = ego_state.get('lane_offset', 0)
        if abs(lane_offset) > 1.5:  # Very close to edge
            risk = 0.8
        elif abs(lane_offset) > 1.0:
            risk = 0.5
        elif abs(lane_offset) > 0.5:
            risk = 0.2
        
        # Check heading relative to road
        heading_error = ego_state.get('heading_error', 0)
        if abs(heading_error) > 0.3:  # Large heading error
            risk = max(risk, 0.6)
        
        return min(1.0, risk)
    
    def _calculate_speed_risk(self, ego_state: Dict, road_conditions: Dict) -> float:
        """Calculate speed-related risk"""
        
        risk = 0.0
        
        speed = ego_state.get('speed', 0)
        speed_limit = road_conditions.get('speed_limit', 15)  # m/s default
        
        if speed > speed_limit + 10:  # More than 10 m/s over limit
            risk = 0.8
        elif speed > speed_limit + 5:
            risk = 0.5
        elif speed > speed_limit:
            risk = 0.2
        
        return min(1.0, risk)
    
    def _determine_safety_level(self, risk_score: float) -> SafetyLevel:
        """Determine safety level from risk score"""
        
        if risk_score >= 0.8:
            return SafetyLevel.EMERGENCY
        elif risk_score >= 0.6:
            return SafetyLevel.CRITICAL
        elif risk_score >= 0.4:
            return SafetyLevel.WARNING
        elif risk_score >= 0.2:
            return SafetyLevel.CAUTION
        else:
            return SafetyLevel.SAFE
    
    def _determine_safety_action(self, safety_level: SafetyLevel, risk_score: float) -> SafetyAction:
        """Determine recommended safety action"""
        
        if safety_level == SafetyLevel.EMERGENCY:
            return SafetyAction.IMMEDIATE_BRAKE
        elif safety_level == SafetyLevel.CRITICAL:
            return SafetyAction.EMERGENCY_STOP
        elif safety_level == SafetyLevel.WARNING:
            return SafetyAction.PREPARE_TO_STOP
        elif safety_level == SafetyLevel.CAUTION:
            return SafetyAction.REDUCE_SPEED
        else:
            return SafetyAction.CONTINUE
    
    def _check_constraints(self, ego_state: Dict, surrounding_objects: List[Dict]) -> List[str]:
        """Check safety constraints and return violations"""
        
        violations = []
        
        # Check acceleration constraints
        acceleration = ego_state.get('acceleration', 0)
        if abs(acceleration) > self.safety_thresholds['max_acceleration']:
            violations.append(f"Acceleration limit exceeded: {acceleration:.2f} m/s²")
        
        # Check deceleration constraints
        if acceleration < -self.safety_thresholds['max_deceleration']:
            violations.append(f"Deceleration limit exceeded: {acceleration:.2f} m/s²")
        
        # Check lateral acceleration
        lateral_accel = ego_state.get('lateral_acceleration', 0)
        if abs(lateral_accel) > self.safety_thresholds['max_lateral_acceleration']:
            violations.append(f"Lateral acceleration limit exceeded: {lateral_accel:.2f} m/s²")
        
        # Check following distance
        min_ttc = float('inf')
        for obj in surrounding_objects:
            ttc = obj.get('ttc', float('inf'))
            if ttc < min_ttc:
                min_ttc = ttc
        
        if min_ttc < self.safety_thresholds['min_ttc']:
            violations.append(f"Time to collision too low: {min_ttc:.2f} s")
        
        return violations
    
    def _generate_safety_reasoning(self, safety_level: SafetyLevel, risk_score: float, 
                                 constraints: List[str]) -> str:
        """Generate safety reasoning explanation"""
        
        reasoning = f"Safety assessment: {safety_level.value} (risk score: {risk_score:.2f})"
        
        if constraints:
            reasoning += f". Constraint violations: {len(constraints)}"
        
        # Add specific reasoning based on safety level
        if safety_level == SafetyLevel.EMERGENCY:
            reasoning += ". Immediate action required to prevent collision"
        elif safety_level == SafetyLevel.CRITICAL:
            reasoning += ". High risk situation detected, prepare for emergency action"
        elif safety_level == SafetyLevel.WARNING:
            reasoning += ". Elevated risk, increase caution"
        elif safety_level == SafetyLevel.CAUTION:
            reasoning += ". Minor risk detected, monitor situation"
        else:
            reasoning += ". Normal driving conditions"
        
        return reasoning
    
    def _check_emergency_conditions(self, ego_state: Dict, surrounding_objects: List[Dict]) -> bool:
        """Check if emergency stop is required"""
        
        # Check for imminent collision
        for obj in surrounding_objects:
            ttc = obj.get('ttc', float('inf'))
            distance = obj.get('distance', float('inf'))
            
            if ttc < 1.0 or distance < 2.0:  # Less than 1 second or 2 meters
                return True
        
        # Check for loss of control
        lateral_accel = ego_state.get('lateral_acceleration', 0)
        yaw_rate = ego_state.get('yaw_rate', 0)
        
        if abs(lateral_accel) > 6.0 or abs(yaw_rate) > 0.8:
            return True
        
        return False
    
    def _emergency_assessment(self) -> SafetyAssessment:
        """Generate emergency assessment on system failure"""
        
        return SafetyAssessment(
            safety_level=SafetyLevel.EMERGENCY,
            risk_score=1.0,
            recommended_action=SafetyAction.IMMEDIATE_BRAKE,
            constraints=["System failure - emergency mode"],
            reasoning="Safety system failure - emergency stop required",
            response_time_ms=1.0,
            emergency_stop_required=True
        )
    
    def _update_metrics(self, assessment: SafetyAssessment):
        """Update performance metrics"""
        
        self.metrics['total_assessments'] += 1
        
        # Update average response time
        current_avg = self.metrics['average_response_time']
        total = self.metrics['total_assessments']
        self.metrics['average_response_time'] = (
            (current_avg * (total - 1) + assessment.response_time_ms) / total
        )
        
        # Update max response time
        self.metrics['max_response_time'] = max(
            self.metrics['max_response_time'], assessment.response_time_ms
        )
        
        # Update emergency activations
        if assessment.emergency_stop_required:
            self.metrics['emergency_activations'] += 1
        
        # Update constraint violations
        if assessment.constraints:
            self.metrics['constraint_violations'] += len(assessment.constraints)
    
    def _store_assessment(self, assessment: SafetyAssessment):
        """Store assessment in history"""
        
        self.assessment_history.append(assessment)
        
        # Limit history size
        if len(self.assessment_history) > self.max_history_size:
            self.assessment_history.pop(0)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def get_recent_assessments(self, count: int = 10) -> List[SafetyAssessment]:
        """Get recent safety assessments"""
        return self.assessment_history[-count:]
    
    def shutdown(self):
        """Shutdown safety critic"""
        logger.info("Shutting down Safety Critic...")
        
        # Clear assessment history
        self.assessment_history.clear()
        
        logger.info("Safety Critic shutdown complete")
