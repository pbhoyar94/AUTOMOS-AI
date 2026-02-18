# CARLA Setup for AUTOMOS AI

## Overview
CARLA is an open-source simulator for autonomous driving research. This guide shows how to set up CARLA for AUTOMOS AI testing.

## Prerequisites
- Python 3.7+
- GPU with OpenGL support (recommended)
- 16GB+ RAM (recommended)
- Unreal Engine 4.24+ (for building from source)

## Installation Options

### Option 1: Download Pre-built Binaries (Recommended)

1. **Download CARLA:**
   ```bash
   # Windows
   wget http://carla-releases.s3.amazonaws.com/Windows/Dev/CARLA_0.9.15.zip
   
   # Linux
   wget http://carla-releases.s3.amazonaws.com/Linux/Dev/CARLA_0.9.15.tar.gz
   ```

2. **Extract:**
   ```bash
   # Windows
   unzip CARLA_0.9.15.zip -d carla
   
   # Linux
   tar -xzf CARLA_0.9.15.tar.gz
   mv CARLA_0.9.15 carla
   ```

### Option 2: Build from Source

1. **Clone CARLA:**
   ```bash
   git clone https://github.com/carla-simulator/carla.git
   cd carla
   git checkout 0.9.15
   ```

2. **Build CARLA:**
   ```bash
   # Windows
   ./Update.bat
   
   # Linux
   ./Update.sh
   make PythonAPI
   ```

## Setup AUTOMOS AI Integration

### 1. Install Dependencies
```bash
pip install numpy networkx distro shapely
```

### 2. Test CARLA Integration
```bash
python carla_integration.py
```

## Running CARLA Server

### Method 1: Using the Start Script
```bash
python start_carla_server.py
```

### Method 2: Manual Start
```bash
# Windows
cd carla/WindowsNoEditor
CarlaUE4.exe -windowed -carla-server -world-port=2000

# Linux
cd carla/LinuxNoEditor
./CarlaUE4.sh -windowed -carla-server -world-port=2000
```

## AUTOMOS AI CARLA Features

### Supported Sensors
- **RGB Camera** - Front-facing camera for visual perception
- **LiDAR** - 3D point cloud sensing
- **Radar** - All-weather object detection

### Integration Capabilities
- Real-time sensor data processing
- AUTOMOS AI control loop integration
- Vehicle state synchronization
- Multi-sensor fusion testing

## Usage Example

```python
from carla_integration import CarlaIntegration

# Create CARLA integration
carla_sim = CarlaIntegration(host='localhost', port=2000)

# Connect and setup
if carla_sim.connect():
    carla_sim.setup_vehicle('vehicle.tesla.model3')
    carla_sim.setup_sensors()
    carla_sim.integrate_automos_ai()
    
    # Run simulation
    carla_sim.run_simulation(duration=60.0)
```

## Troubleshooting

### Common Issues

1. **CARLA not found**
   - Ensure CARLA is properly installed
   - Check the CARLA_ROOT path in integration script

2. **Connection failed**
   - Make sure CARLA server is running
   - Check firewall settings
   - Verify port 2000 is available

3. **Sensor setup failed**
   - Ensure vehicle is spawned first
   - Check blueprint availability

4. **Performance issues**
   - Reduce quality settings
   - Disable unnecessary sensors
   - Use smaller map

### Performance Optimization

1. **CARLA Server Settings:**
   ```bash
   CarlaUE4.exe -quality-level=Low -no-rendering-offscreen
   ```

2. **Sensor Configuration:**
   - Reduce camera resolution
   - Lower LiDAR point density
   - Adjust update rates

## Development

### Adding New Sensors
1. Add sensor setup in `setup_sensors()`
2. Implement mock data generation
3. Update AUTOMOS AI integration

### Custom Scenarios
1. Create scenario scripts
2. Use CARLA Scenario Runner
3. Integrate with AUTOMOS AI testing

## Resources

- [CARLA Documentation](https://carla.readthedocs.io/)
- [CARLA GitHub](https://github.com/carla-simulator/carla)
- [AUTOMOS AI Documentation](./README.md)

## Support

For CARLA-specific issues, visit:
- [CARLA Forum](https://github.com/carla-simulator/carla/discussions)
- [CARLA Discord](https://discord.gg/8kqACuC)

For AUTOMOS AI integration issues, check the integration logs and ensure all dependencies are properly installed.
