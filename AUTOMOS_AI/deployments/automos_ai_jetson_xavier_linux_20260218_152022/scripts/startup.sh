#!/bin/bash
# AUTOMOS AI Platform Startup Script
# Platform: jetson_xavier

echo "Starting AUTOMOS AI on jetson_xavier..."

# Platform-specific initialization
if [ "jetson_xavier" = "jetson_nano" ]; then
    # Jetson Nano specific setup
    nvpmodel -m 0  # Max power mode
    jetson_clocks  # Max clocks
elif [ "jetson_xavier" = "jetson_xavier" ]; then
    # Jetson Xavier specific setup
    nvpmodel -m 0  # Max power mode
    jetson_clocks  # Max clocks
elif [ "jetson_xavier" = "jetson_orin" ]; then
    # Jetson Orin specific setup
    nvpmodel -m 0  # Max power mode
    jetson_clocks  # Max clocks
fi

# Start AUTOMOS AI
/opt/automos_ai/bin/automos_ai "$@"
