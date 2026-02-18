#!/usr/bin/env python3
"""
AUTOMOS AI - CARLA Integration
Integration with CARLA simulator for autonomous driving testing
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add CARLA Python API to path
CARLA_ROOT = Path(__file__).parent.parent / "carla" / "PythonAPI" / "carla"
if CARLA_ROOT.exists():
    sys.path.insert(0, str(CARLA_ROOT))

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("Warning: CARLA not available. Please install CARLA simulator.")

logger = logging.getLogger(__name__)

class CarlaIntegration:
    """CARLA simulator integration for AUTOMOS AI testing"""
    
    def __init__(self, host: str = 'localhost', port: int = 2000, timeout: float = 10.0):
        """
        Initialize CARLA integration
        
        Args:
            host: CARLA server host
            port: CARLA server port
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        # CARLA objects
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensors = {}
        
        # AUTOMOS AI integration
        self.automos_ai = None
        
        logger.info(f"CARLA Integration initialized for {host}:{port}")
    
    def connect(self) -> bool:
        """
        Connect to CARLA server
        
        Returns:
            True if connection successful, False otherwise
        """
        if not CARLA_AVAILABLE:
            logger.error("CARLA not available. Please install CARLA simulator.")
            return False
        
        try:
            logger.info(f"Connecting to CARLA server at {self.host}:{self.port}...")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            
            # Test connection
            self.world = self.client.get_world()
            logger.info("Connected to CARLA server successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False
    
    def setup_vehicle(self, vehicle_filter: str = 'vehicle.tesla.model3') -> bool:
        """
        Setup ego vehicle in CARLA
        
        Args:
            vehicle_filter: Vehicle blueprint filter
            
        Returns:
            True if vehicle setup successful, False otherwise
        """
        if not self.world:
            logger.error("CARLA world not available")
            return False
        
        try:
            # Get vehicle blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter(vehicle_filter)[0]
            
            # Set vehicle color
            if vehicle_bp.has_attribute('color'):
                color = vehicle_bp.get_attribute('color').recommended_values[0]
                vehicle_bp.set_attribute('color', color)
            
            # Get spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                logger.error("No spawn points available")
                return False
            
            spawn_point = spawn_points[0]
            
            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            logger.info(f"Vehicle spawned: {vehicle_filter}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup vehicle: {e}")
            return False
    
    def setup_sensors(self) -> bool:
        """
        Setup sensors for AUTOMOS AI perception
        
        Returns:
            True if sensor setup successful, False otherwise
        """
        if not self.vehicle:
            logger.error("Vehicle not available")
            return False
        
        try:
            blueprint_library = self.world.get_blueprint_library()
            
            # Setup RGB camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            
            camera_transform = carla.Transform(
                carla.Location(x=1.5, z=2.4),
                carla.Rotation(pitch=0)
            )
            
            camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            self.sensors['camera'] = camera
            
            # Setup LiDAR
            lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels', '32')
            lidar_bp.set_attribute('points_per_second', '90000')
            lidar_bp.set_attribute('rotation_frequency', '10')
            lidar_bp.set_attribute('range', '100')
            
            lidar_transform = carla.Transform(
                carla.Location(x=0, z=2.5),
                carla.Rotation(pitch=0)
            )
            
            lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
            self.sensors['lidar'] = lidar
            
            # Setup radar
            radar_bp = blueprint_library.find('sensor.other.radar')
            radar_bp.set_attribute('horizontal_fov', '70')
            radar_bp.set_attribute('vertical_fov', '30')
            radar_bp.set_attribute('points_per_second', '1500')
            radar_bp.set_attribute('range', '100')
            
            radar_transform = carla.Transform(
                carla.Location(x=2.5, z=0.5),
                carla.Rotation(pitch=0)
            )
            
            radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.vehicle)
            self.sensors['radar'] = radar
            
            logger.info("Sensors setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup sensors: {e}")
            return False
    
    def integrate_automos_ai(self) -> bool:
        """
        Integrate AUTOMOS AI with CARLA
        
        Returns:
            True if integration successful, False otherwise
        """
        try:
            # Import AUTOMOS AI
            from main import AUTOMOS_AI
            
            # Initialize AUTOMOS AI
            self.automos_ai = AUTOMOS_AI()
            
            logger.info("AUTOMOS AI integrated with CARLA")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate AUTOMOS AI: {e}")
            return False
    
    def run_simulation(self, duration: float = 60.0) -> bool:
        """
        Run simulation with AUTOMOS AI control
        
        Args:
            duration: Simulation duration in seconds
            
        Returns:
            True if simulation successful, False otherwise
        """
        if not self.vehicle or not self.automos_ai:
            logger.error("Vehicle or AUTOMOS AI not available")
            return False
        
        try:
            logger.info(f"Starting simulation for {duration} seconds...")
            
            start_time = time.time()
            simulation_running = True
            
            while simulation_running and (time.time() - start_time) < duration:
                # Get vehicle state
                transform = self.vehicle.get_transform()
                velocity = self.vehicle.get_velocity()
                
                # Create sensor data mock for AUTOMOS AI
                sensor_data = {
                    'camera': self._get_mock_camera_data(),
                    'lidar': self._get_mock_lidar_data(),
                    'radar': self._get_mock_radar_data()
                }
                
                # Process with AUTOMOS AI
                if hasattr(self.automos_ai, 'process_sensor_data'):
                    control_output = self.automos_ai.process_sensor_data(sensor_data)
                    
                    # Apply control to vehicle
                    if control_output:
                        self._apply_vehicle_control(control_output)
                
                # Step simulation
                self.world.tick()
                time.sleep(0.1)  # 10 Hz update rate
            
            logger.info("Simulation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return False
    
    def _get_mock_camera_data(self) -> np.ndarray:
        """Get mock camera data for AUTOMOS AI"""
        return np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    
    def _get_mock_lidar_data(self) -> np.ndarray:
        """Get mock LiDAR data for AUTOMOS AI"""
        return np.random.rand(1000, 3).astype(np.float32)
    
    def _get_mock_radar_data(self) -> np.ndarray:
        """Get mock radar data for AUTOMOS AI"""
        return np.random.rand(100, 4).astype(np.float32)
    
    def _apply_vehicle_control(self, control_output: Dict):
        """Apply control output to CARLA vehicle"""
        try:
            control = carla.VehicleControl()
            
            if 'throttle' in control_output:
                control.throttle = float(control_output['throttle'])
            if 'steer' in control_output:
                control.steer = float(control_output['steer'])
            if 'brake' in control_output:
                control.brake = float(control_output['brake'])
            
            self.vehicle.apply_control(control)
            
        except Exception as e:
            logger.error(f"Failed to apply vehicle control: {e}")
    
    def cleanup(self):
        """Cleanup CARLA resources"""
        try:
            # Destroy sensors
            for sensor in self.sensors.values():
                if sensor and sensor.is_alive:
                    sensor.destroy()
            
            # Destroy vehicle
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy()
            
            logger.info("CARLA resources cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

def test_carla_integration():
    """Test CARLA integration"""
    logging.basicConfig(level=logging.INFO)
    
    with CarlaIntegration() as carla_sim:
        if carla_sim.connect():
            if carla_sim.setup_vehicle():
                if carla_sim.setup_sensors():
                    if carla_sim.integrate_automos_ai():
                        carla_sim.run_simulation(duration=10.0)
                    else:
                        logger.error("Failed to integrate AUTOMOS AI")
                else:
                    logger.error("Failed to setup sensors")
            else:
                logger.error("Failed to setup vehicle")
        else:
            logger.error("Failed to connect to CARLA")

if __name__ == "__main__":
    test_carla_integration()
