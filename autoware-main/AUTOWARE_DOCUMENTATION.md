# Autoware - Comprehensive Documentation

## Overview

Autoware is the world's leading open-source software project for autonomous driving, built on the Robot Operating System (ROS). This meta-repository serves as the central entry point for accessing and managing the complete Autoware ecosystem.

## What is Autoware?

Autoware is a comprehensive autonomous driving software stack that includes all necessary functions for self-driving vehicles:

- **Localization**: Determining the vehicle's position in the world
- **Object Detection**: Identifying and tracking objects around the vehicle
- **Route Planning**: Computing optimal paths from current location to destination
- **Vehicle Control**: Managing steering, acceleration, and braking
- **Sensor Integration**: Processing data from LiDAR, cameras, radar, and GPS

## Repository Structure

This is a **meta-repository** that manages multiple Git repositories using the `vcs` (version control system) tool. The actual source code is distributed across several repositories:

### Core Components
- **autoware_msgs**: Message definitions for Autoware communication
- **autoware_adapi_msgs**: API message definitions
- **autoware_internal_msgs**: Internal message types
- **autoware_cmake**: CMake utilities for building
- **autoware_utils**: Utility functions and helpers
- **autoware_lanelet2_extension**: Lanelet2 map format extensions
- **autoware_core**: Core autonomous driving modules
- **autoware_rviz_plugins**: Visualization plugins for RViz

### Universe Components (Experimental/Advanced)
- **autoware_universe**: Cutting-edge experimental features
- **External Dependencies**: Various third-party integrations including:
  - MORAI simulation messages
  - Point cloud processing
  - GNSS/RTK libraries
  - TensorRT optimizations
  - CUDA acceleration modules

### Launch Configuration
- **autoware_launch**: Launch files and parameter configurations

### Sensor Components
- **sensor_component_description**: Sensor component descriptions and configurations

## Key Files and Their Functions

### setup-dev-env.sh
**Purpose**: Development environment setup script
**Functionality**:
- Installs ROS 2 (Humble or Jazzy)
- Sets up Ansible for automated deployment
- Configures NVIDIA CUDA/TensorRT support
- Downloads necessary dependencies and artifacts
- Creates development environment with all required tools

**Usage**:
```bash
./setup-dev-env.sh [core|universe] [options]
```

**Options**:
- `-y`: Non-interactive mode
- `-v`: Verbose output
- `--no-nvidia`: Skip NVIDIA components
- `--data-dir`: Set custom data directory
- `--ros-distro`: Choose ROS distribution (humble/jazzy)

### repositories/autoware.repos
**Purpose**: Manifest file defining all Autoware repositories and their versions
**Function**: Used by `vcs import` to clone the correct versions of all source repositories

### src/ Directory
**Purpose**: Target directory for all source code repositories
**Note**: Initially empty, populated by running `vcs import` command

## Installation and Setup

### Prerequisites
- Ubuntu 20.04+ or compatible Linux distribution
- Git
- Python 3
- Sudo access

### Setup Process
1. Clone this meta-repository
2. Run the development environment setup script
3. Import all source repositories using vcs
4. Build the workspace using colcon

### Commands
```bash
# Clone the meta-repository
git clone https://github.com/autowarefoundation/autoware.git
cd autoware

# Set up development environment
./setup-dev-env.sh universe

# Import source repositories
vcs import --recursive src < repositories/autoware.repos

# Build the workspace
colcon build --symlink-install
```

## Architecture

Autoware follows a modular architecture with clear separation of concerns:

### Perception Layer
- Sensor data processing
- Object detection and tracking
- Environment mapping

### Planning Layer
- Mission planning (high-level route planning)
- Behavior planning (decision making)
- Motion planning (trajectory generation)

### Control Layer
- Vehicle control algorithms
- Actuator command generation
- Safety monitoring

### Localization Layer
- GNSS/INS integration
- LiDAR-based localization
- Sensor fusion

## Key Features

### Multi-Sensor Support
- LiDAR point cloud processing
- Camera-based perception
- Radar object detection
- GNSS/RTK positioning

### Advanced Algorithms
- SLAM (Simultaneous Localization and Mapping)
- Path planning using A*, RRT*, and other algorithms
- Model predictive control
- Deep learning-based object detection

### Simulation Integration
- CARLA simulator support
- LGSVL simulator compatibility
- Custom simulation frameworks

### Real-time Capabilities
- ROS 2 real-time extensions
- Deterministic processing pipelines
- Hardware acceleration support

## Development Workflow

### For Developers
1. Fork the relevant repository
2. Create feature branches
3. Make changes following coding standards
4. Submit pull requests
5. Participate in code reviews

### For Users
1. Set up development environment
2. Configure vehicle parameters
3. Calibrate sensors
4. Test in simulation
5. Deploy to vehicle

## Community and Support

### Resources
- Official Documentation: https://autowarefoundation.github.io/autoware-documentation/
- GitHub Discussions: Q&A and community support
- Discord: Real-time chat and support
- Working Groups: Specialized development teams

### Contributing
- No formal contributor process required
- Follow contribution guidelines
- Respect code of conduct
- Participate in working groups

## Versions and Releases

### Autoware Core
- Stable, production-ready components
- Thoroughly tested modules
- Long-term support

### Autoware Universe
- Experimental features
- Cutting-edge algorithms
- Research-oriented components

### ROS 2 Support
- Humble (LTS): Default stable version
- Jazzy: Latest features

## Licensing

- Apache 2.0 License
- Permissive for commercial and academic use
- Patent protection included

## System Requirements

### Minimum Requirements
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GTX 1060+ (for perception modules)
- Storage: 100GB+ SSD

### Recommended Requirements
- CPU: 16+ cores
- RAM: 32GB+
- GPU: NVIDIA RTX 3080+ or equivalent
- Storage: 500GB+ NVMe SSD

## Conclusion

Autoware represents the most comprehensive open-source autonomous driving platform available today. Its modular architecture, extensive feature set, and active community make it suitable for both research and production applications. The meta-repository structure ensures easy management of the complex ecosystem while maintaining version compatibility across all components.

This documentation provides a foundation for understanding the Autoware ecosystem. For detailed implementation guides and API references, consult the official documentation and individual repository README files.
