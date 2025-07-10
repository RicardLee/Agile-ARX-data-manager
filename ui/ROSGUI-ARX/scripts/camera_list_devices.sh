#!/bin/bash

# Camera: List Devices
# Based on commands_to_use.md section 1

echo "Starting camera device listing..."

# Launch astra camera list devices
echo "Launching astra_camera list_devices.launch..."
roslaunch astra_camera list_devices.launch
