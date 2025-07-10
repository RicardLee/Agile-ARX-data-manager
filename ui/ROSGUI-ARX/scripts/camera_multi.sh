#!/bin/bash

# Camera: Multi Camera Launch
# Based on commands_to_use.md section 1

echo "Starting multi-camera setup..."

# Launch astra camera multi camera
echo "Launching astra_camera multi_camera.launch..."
roslaunch astra_camera multi_camera_455.launch
