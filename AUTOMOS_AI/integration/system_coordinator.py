"""
AUTOMOS AI System Coordinator
Coordinates all system components and manages execution flow
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"

class ExecutionMode(Enum):
    """Execution modes"""
    SIMULATION = "simulation"
    DEPLOYMENT = "deployment"
    TESTING = "testing"
    DEBUG = "debug"

@dataclass
class SystemStatus:
    """System status information"""
    state: SystemState
    execution_mode: ExecutionMode
    uptime: float
    component_status: Dict[str, bool]
    performance_metrics: Dict[str, float]
    error_count: int
    last_update: float

@dataclass
class DrivingCommand:
    """Driving command for vehicle control"""
    steering_angle: float
    acceleration: float
    brake_pressure: float
    target_speed: float
    gear: int
    timestamp: float
    confidence: float

class SystemCoordinator:
    """Main system coordinator for AUTOMOS AI"""
    
    def __init__(self):
        """Initialize system coordinator"""
        logger.info("Initializing System Coordinator...")
        
        # System state
        self.state = SystemState.INITIALIZING
        self.execution_mode = ExecutionMode.SIMULATION
        self.start_time = time.time()
        
        # Component references
        self.reasoning_engine = None
        self.perception_pipeline = None
        self.safety_critic = None
        self.world_model = None
        
        # Control interfaces
        self.vehicle_interface = None
        self.hmi_interface = None
        
        # Communication
        self.command_queue = queue.Queue(maxsize=100)
        self.status_queue = queue.Queue(maxsize=10)
        self.emergency_event = threading.Event()
        
        # Execution control
        self.execution_thread = None
        self.is_running = False
        self.execution_frequency = 20.0  # Hz
        
        # Performance tracking
        self.metrics = {
            'total_cycles': 0,
            'average_cycle_time': 0.0,
            'successful_plans': 0,
            'emergency_stops': 0,
            'component_failures': 0
        }
        
        # Error handling
        self.error_history = []
        self.max_error_history = 100
        
        logger.info("System Coordinator initialized")
    
    def start(self):
        """Start system coordinator"""
        logger.info("Starting System Coordinator...")
        
        try:
            self.state = SystemState.RUNNING
            self.is_running = True
            
            # Start execution thread
            self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
            self.execution_thread.start()
            
            # Start status monitoring
            status_thread = threading.Thread(target=self._status_monitoring_loop, daemon=True)
            status_thread.start()
            
            logger.info("System Coordinator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start System Coordinator: {e}")
            self.state = SystemState.ERROR
            raise
    
    def set_components(self, reasoning_engine, perception_pipeline, safety_critic, world_model):
        """Set component references"""
        
        self.reasoning_engine = reasoning_engine
        self.perception_pipeline = perception_pipeline
        self.safety_critic = safety_critic
        self.world_model = world_model
        
        logger.info("Component references set")
    
    def execute_plan(self, driving_plan):
        """
        Execute driving plan
        
        Args:
            driving_plan: Driving plan from reasoning engine
        """
        
        try:
            # Convert plan to driving commands
            commands = self._plan_to_commands(driving_plan)
            
            # Send commands to vehicle interface
            for command in commands:
                self.command_queue.put(command)
            
            logger.debug(f"Executed driving plan with {len(commands)} commands")
            
        except Exception as e:
            logger.error(f"Failed to execute plan: {e}")
            self._handle_error("Plan execution failed", e)
    
    def emergency_stop(self):
        """Execute emergency stop"""
        logger.critical("EMERGENCY STOP ACTIVATED")
        
        self.state = SystemState.EMERGENCY_STOP
        self.emergency_event.set()
        self.metrics['emergency_stops'] += 1
        
        # Create emergency stop command
        emergency_command = DrivingCommand(
            steering_angle=0.0,
            acceleration=0.0,
            brake_pressure=1.0,  # Full brake
            target_speed=0.0,
            gear=0,  # Neutral
            timestamp=time.time(),
            confidence=1.0
        )
        
        # Send emergency command
        self.command_queue.put(emergency_command)
        
        # Clear command queue of other commands
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break
    
    def _execution_loop(self):
        """Main execution loop"""
        
        logger.info("Starting execution loop")
        
        while self.is_running:
            cycle_start = time.time()
            
            try:
                # Check for emergency conditions
                if self.emergency_event.is_set():
                    self._handle_emergency_state()
                    continue
                
                # Execute system cycle
                self._execute_system_cycle()
                
                # Update metrics
                cycle_time = (time.time() - cycle_start) * 1000
                self._update_cycle_metrics(cycle_time)
                
                # Sleep to maintain frequency
                sleep_time = max(0, (1.0 / self.execution_frequency) - (time.time() - cycle_start))
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                self._handle_error("Execution loop error", e)
                time.sleep(0.1)
        
        logger.info("Execution loop ended")
    
    def _execute_system_cycle(self):
        """Execute one system cycle"""
        
        # 1. Get perception data
        perception_result = self.perception_pipeline.process_sensors()
        
        # 2. Update world model
        world_state = self.world_model.update(perception_result._asdict())
        
        # 3. Safety evaluation
        safety_assessment = self.safety_critic.evaluate(world_state._asdict())
        
        # 4. Check for emergency conditions
        if safety_assessment.emergency_stop_required:
            self.emergency_stop()
            return
        
        # 5. Generate driving plan
        driving_plan = self.reasoning_engine.generate_plan(
            world_state=world_state,
            safety_constraints=safety_assessment.constraints,
            language_command="Navigate safely"
        )
        
        # 6. Execute plan
        self.execute_plan(driving_plan)
        
        # 7. Update status
        self._update_system_status()
        
        self.metrics['successful_plans'] += 1
    
    def _plan_to_commands(self, driving_plan) -> List[DrivingCommand]:
        """Convert driving plan to vehicle commands"""
        
        commands = []
        
        try:
            # Extract plan data
            trajectory = driving_plan.trajectory
            velocity_profile = driving_plan.velocity_profile
            steering_commands = driving_plan.steering_commands
            
            # Generate commands for each timestep
            for i in range(len(trajectory)):
                if i >= len(velocity_profile) or i >= len(steering_commands):
                    break
                
                command = DrivingCommand(
                    steering_angle=steering_commands[i],
                    acceleration=0.0,  # Will be calculated from velocity profile
                    brake_pressure=0.0,
                    target_speed=velocity_profile[i],
                    gear=1,  # Drive
                    timestamp=time.time() + i * 0.1,  # 100ms intervals
                    confidence=driving_plan.confidence_score
                )
                
                # Calculate acceleration from velocity profile
                if i > 0:
                    dv = velocity_profile[i] - velocity_profile[i-1]
                    dt = 0.1
                    command.acceleration = dv / dt
                    
                    # Convert to brake pressure if decelerating
                    if command.acceleration < 0:
                        command.brake_pressure = min(1.0, abs(command.acceleration) / 8.0)  # Max 8 m/s² deceleration
                        command.acceleration = 0.0
                
                commands.append(command)
                
                # Limit command queue size
                if len(commands) >= 50:  # 5 seconds of commands
                    break
            
        except Exception as e:
            logger.error(f"Failed to convert plan to commands: {e}")
            # Return safe default command
            commands = [DrivingCommand(
                steering_angle=0.0, acceleration=0.0, brake_pressure=0.0,
                target_speed=0.0, gear=0, timestamp=time.time(), confidence=0.5
            )]
        
        return commands
    
    def _handle_emergency_state(self):
        """Handle emergency state"""
        
        logger.warning("Handling emergency state")
        
        # Send emergency stop command
        emergency_command = DrivingCommand(
            steering_angle=0.0,
            acceleration=0.0,
            brake_pressure=1.0,
            target_speed=0.0,
            gear=0,
            timestamp=time.time(),
            confidence=1.0
        )
        
        self.command_queue.put(emergency_command)
        
        # Wait for emergency to be cleared
        time.sleep(0.1)
    
    def _status_monitoring_loop(self):
        """Status monitoring loop"""
        
        while self.is_running:
            try:
                # Generate system status
                status = self._generate_system_status()
                
                # Add to status queue
                try:
                    self.status_queue.put_nowait(status)
                except queue.Full:
                    # Remove old status
                    try:
                        self.status_queue.get_nowait()
                        self.status_queue.put_nowait(status)
                    except queue.Empty:
                        pass
                
                # Sleep for monitoring frequency
                time.sleep(1.0)  # 1 Hz monitoring
                
            except Exception as e:
                logger.error(f"Error in status monitoring: {e}")
                time.sleep(1.0)
    
    def _generate_system_status(self) -> SystemStatus:
        """Generate current system status"""
        
        # Check component status
        component_status = {
            'reasoning_engine': self.reasoning_engine is not None,
            'perception_pipeline': self.perception_pipeline is not None,
            'safety_critic': self.safety_critic is not None,
            'world_model': self.world_model is not None
        }
        
        # Get performance metrics
        performance_metrics = {
            'cycle_time': self.metrics['average_cycle_time'],
            'success_rate': (self.metrics['successful_plans'] / max(1, self.metrics['total_cycles'])) * 100,
            'emergency_stops': self.metrics['emergency_stops'],
            'component_failures': self.metrics['component_failures']
        }
        
        return SystemStatus(
            state=self.state,
            execution_mode=self.execution_mode,
            uptime=time.time() - self.start_time,
            component_status=component_status,
            performance_metrics=performance_metrics,
            error_count=len(self.error_history),
            last_update=time.time()
        )
    
    def _update_system_status(self):
        """Update system status"""
        
        # Check component health
        if not all([
            self.reasoning_engine,
            self.perception_pipeline,
            self.safety_critic,
            self.world_model
        ]):
            self.metrics['component_failures'] += 1
            logger.warning("One or more components are not available")
    
    def _update_cycle_metrics(self, cycle_time: float):
        """Update cycle performance metrics"""
        
        self.metrics['total_cycles'] += 1
        
        # Update average cycle time
        current_avg = self.metrics['average_cycle_time']
        total = self.metrics['total_cycles']
        self.metrics['average_cycle_time'] = (current_avg * (total - 1) + cycle_time) / total
    
    def _handle_error(self, message: str, error: Exception):
        """Handle system error"""
        
        error_info = {
            'message': message,
            'error': str(error),
            'timestamp': time.time(),
            'state': self.state.value
        }
        
        self.error_history.append(error_info)
        
        # Limit error history size
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        logger.error(f"System error: {message} - {error}")
        
        # Check if we need to enter error state
        if len(self.error_history) > 10:  # Too many errors
            self.state = SystemState.ERROR
    
    def get_status(self) -> Optional[SystemStatus]:
        """Get current system status"""
        
        try:
            return self.status_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def get_error_history(self, count: int = 10) -> List[Dict]:
        """Get recent error history"""
        return self.error_history[-count:]
    
    def shutdown(self):
        """Shutdown system coordinator"""
        logger.info("Shutting down System Coordinator...")
        
        self.state = SystemState.SHUTTING_DOWN
        self.is_running = False
        
        # Wait for execution thread to finish
        if self.execution_thread:
            self.execution_thread.join(timeout=2.0)
        
        # Clear queues
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.status_queue.empty():
            try:
                self.status_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear emergency event
        self.emergency_event.clear()
        
        logger.info("System Coordinator shutdown complete")
