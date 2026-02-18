"""
AUTOMOS AI System Monitor
Real-time system monitoring for production deployment
"""

import os
import logging
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System metrics data structure"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    temperature: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    network_io: Optional[Dict] = None

@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: float
    processing_fps: float
    inference_time_ms: float
    safety_response_time_ms: float
    objects_detected: int
    emergency_events: int
    system_uptime: float

class SystemMonitor:
    """Real-time system monitoring"""
    
    def __init__(self):
        """Initialize system monitor"""
        logger.info("Initializing System Monitor...")
        
        # Monitoring configuration
        self.monitoring_interval = 1.0  # seconds
        self.metrics_history_size = 1000
        
        # Metrics storage
        self.system_metrics_history = []
        self.application_metrics_history = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Thresholds for alerts
        self.thresholds = {
            'cpu_usage': 80.0,  # percentage
            'memory_usage': 85.0,  # percentage
            'disk_usage': 90.0,  # percentage
            'temperature': 75.0,  # celsius
            'gpu_usage': 90.0,  # percentage
            'processing_fps': 15.0,  # minimum FPS
            'safety_response_time': 15.0,  # milliseconds
            'inference_time': 100.0  # milliseconds
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        logger.info("System Monitor initialized")
    
    def start_monitoring(self):
        """Start system monitoring"""
        
        if self.is_monitoring:
            logger.warning("System monitoring already started")
            return
        
        logger.info("Starting system monitoring...")
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        
        if not self.is_monitoring:
            logger.warning("System monitoring not running")
            return
        
        logger.info("Stopping system monitoring...")
        
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                self.application_metrics_history.append(app_metrics)
                
                # Check thresholds and trigger alerts
                self._check_thresholds(system_metrics, app_metrics)
                
                # Limit history size
                if len(self.system_metrics_history) > self.metrics_history_size:
                    self.system_metrics_history.pop(0)
                
                if len(self.application_metrics_history) > self.metrics_history_size:
                    self.application_metrics_history.pop(0)
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        
        try:
            # Try to import psutil for system metrics
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Temperature (if available)
            temperature = self._get_system_temperature()
            
            # GPU metrics (if available)
            gpu_usage = None
            gpu_memory = None
            
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                    # GPU usage would require nvidia-ml-py or similar
                    gpu_usage = 0.0  # Placeholder
            except ImportError:
                pass
            
            # Network I/O
            network_io = self._get_network_io()
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                temperature=temperature,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory,
                network_io=network_io
            )
            
        except ImportError:
            # Fallback to mock metrics if psutil not available
            return self._get_mock_system_metrics()
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        
        # These would be collected from the actual AUTOMOS AI components
        # For now, return mock metrics
        
        return ApplicationMetrics(
            timestamp=time.time(),
            processing_fps=20.0,  # Mock 20 FPS
            inference_time_ms=25.0,  # Mock 25ms inference
            safety_response_time_ms=8.0,  # Mock 8ms safety response
            objects_detected=5,  # Mock 5 objects detected
            emergency_events=0,  # Mock no emergency events
            system_uptime=time.time() - getattr(self, 'start_time', time.time())
        )
    
    def _get_system_temperature(self) -> float:
        """Get system temperature"""
        
        try:
            # Try to get temperature from system files (Linux)
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_millidegrees = int(f.read().strip())
                    return temp_millidegrees / 1000.0
            
            # Try other temperature sources
            # This would need to be expanded for different systems
            
        except Exception:
            pass
        
        # Return mock temperature if unable to read
        return 45.0  # Mock 45°C
    
    def _get_network_io(self) -> Dict:
        """Get network I/O statistics"""
        
        try:
            import psutil
            net_io = psutil.net_io_counters()
            
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
        except ImportError:
            return {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
    
    def _get_mock_system_metrics(self) -> SystemMetrics:
        """Get mock system metrics when psutil is not available"""
        
        import random
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_usage=random.uniform(20, 60),
            memory_usage=random.uniform(30, 70),
            disk_usage=random.uniform(40, 80),
            temperature=random.uniform(35, 55),
            gpu_usage=random.uniform(10, 40) if random.random() > 0.5 else None,
            gpu_memory=random.uniform(20, 60) if random.random() > 0.5 else None,
            network_io={'bytes_sent': random.randint(1000, 10000), 'bytes_recv': random.randint(1000, 10000)}
        )
    
    def _check_thresholds(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Check thresholds and trigger alerts"""
        
        alerts = []
        
        # Check system thresholds
        if system_metrics.cpu_usage > self.thresholds['cpu_usage']:
            alerts.append({
                'type': 'system',
                'metric': 'cpu_usage',
                'value': system_metrics.cpu_usage,
                'threshold': self.thresholds['cpu_usage'],
                'severity': 'warning' if system_metrics.cpu_usage < 95 else 'critical',
                'timestamp': system_metrics.timestamp
            })
        
        if system_metrics.memory_usage > self.thresholds['memory_usage']:
            alerts.append({
                'type': 'system',
                'metric': 'memory_usage',
                'value': system_metrics.memory_usage,
                'threshold': self.thresholds['memory_usage'],
                'severity': 'warning' if system_metrics.memory_usage < 95 else 'critical',
                'timestamp': system_metrics.timestamp
            })
        
        if system_metrics.temperature > self.thresholds['temperature']:
            alerts.append({
                'type': 'system',
                'metric': 'temperature',
                'value': system_metrics.temperature,
                'threshold': self.thresholds['temperature'],
                'severity': 'warning' if system_metrics.temperature < 85 else 'critical',
                'timestamp': system_metrics.timestamp
            })
        
        # Check application thresholds
        if app_metrics.processing_fps < self.thresholds['processing_fps']:
            alerts.append({
                'type': 'application',
                'metric': 'processing_fps',
                'value': app_metrics.processing_fps,
                'threshold': self.thresholds['processing_fps'],
                'severity': 'warning',
                'timestamp': app_metrics.timestamp
            })
        
        if app_metrics.safety_response_time_ms > self.thresholds['safety_response_time']:
            alerts.append({
                'type': 'application',
                'metric': 'safety_response_time',
                'value': app_metrics.safety_response_time_ms,
                'threshold': self.thresholds['safety_response_time'],
                'severity': 'critical',
                'timestamp': app_metrics.timestamp
            })
        
        # Trigger alert callbacks
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict):
        """Trigger alert callbacks"""
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback):
        """Add alert callback function"""
        
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict:
        """Get current system and application metrics"""
        
        current_system = self.system_metrics_history[-1] if self.system_metrics_history else None
        current_app = self.application_metrics_history[-1] if self.application_metrics_history else None
        
        return {
            'system_metrics': current_system,
            'application_metrics': current_app,
            'monitoring_active': self.is_monitoring
        }
    
    def get_metrics_history(self, duration_minutes: int = 60) -> Dict:
        """Get metrics history for specified duration"""
        
        cutoff_time = time.time() - (duration_minutes * 60)
        
        system_history = [
            metrics for metrics in self.system_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
        
        app_history = [
            metrics for metrics in self.application_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
        
        return {
            'system_metrics': system_history,
            'application_metrics': app_history,
            'duration_minutes': duration_minutes
        }
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        
        if not self.application_metrics_history:
            return {'error': 'No metrics available'}
        
        # Calculate statistics from recent metrics
        recent_metrics = self.application_metrics_history[-100:]  # Last 100 data points
        
        fps_values = [m.processing_fps for m in recent_metrics]
        inference_times = [m.inference_time_ms for m in recent_metrics]
        safety_times = [m.safety_response_time_ms for m in recent_metrics]
        
        return {
            'performance_summary': {
                'avg_fps': sum(fps_values) / len(fps_values),
                'min_fps': min(fps_values),
                'max_fps': max(fps_values),
                'avg_inference_time_ms': sum(inference_times) / len(inference_times),
                'min_inference_time_ms': min(inference_times),
                'max_inference_time_ms': max(inference_times),
                'avg_safety_response_time_ms': sum(safety_times) / len(safety_times),
                'min_safety_response_time_ms': min(safety_times),
                'max_safety_response_time_ms': max(safety_times),
                'total_objects_detected': sum(m.objects_detected for m in recent_metrics),
                'total_emergency_events': sum(m.emergency_events for m in recent_metrics)
            },
            'system_health': {
                'cpu_usage': self.system_metrics_history[-1].cpu_usage if self.system_metrics_history else 0,
                'memory_usage': self.system_metrics_history[-1].memory_usage if self.system_metrics_history else 0,
                'temperature': self.system_metrics_history[-1].temperature if self.system_metrics_history else 0
            }
        }
    
    def export_metrics(self, filename: str, format: str = 'json'):
        """Export metrics to file"""
        
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'system_metrics_history': [
                    {
                        'timestamp': m.timestamp,
                        'cpu_usage': m.cpu_usage,
                        'memory_usage': m.memory_usage,
                        'disk_usage': m.disk_usage,
                        'temperature': m.temperature,
                        'gpu_usage': m.gpu_usage,
                        'gpu_memory': m.gpu_memory
                    } for m in self.system_metrics_history
                ],
                'application_metrics_history': [
                    {
                        'timestamp': m.timestamp,
                        'processing_fps': m.processing_fps,
                        'inference_time_ms': m.inference_time_ms,
                        'safety_response_time_ms': m.safety_response_time_ms,
                        'objects_detected': m.objects_detected,
                        'emergency_events': m.emergency_events,
                        'system_uptime': m.system_uptime
                    } for m in self.application_metrics_history
                ]
            }
            
            if format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format.lower() == 'csv':
                # Export as CSV (would need pandas or similar)
                logger.warning("CSV export not implemented")
                return False
            
            logger.info(f"Metrics exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False
    
    def set_thresholds(self, thresholds: Dict):
        """Update monitoring thresholds"""
        
        self.thresholds.update(thresholds)
        logger.info(f"Updated thresholds: {thresholds}")
    
    def get_thresholds(self) -> Dict:
        """Get current thresholds"""
        
        return self.thresholds.copy()
