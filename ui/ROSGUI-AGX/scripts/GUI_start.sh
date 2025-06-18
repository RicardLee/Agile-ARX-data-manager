#!/bin/bash

# ========== Fail-safe Design ==========
# Kill any existing ROS processes to ensure clean startup
echo "Checking for existing ROS processes..."

# Get all ROS-related process PIDs
ROSCORE_PIDS=$(pgrep -f "roscore" || true)
ROSMASTER_PIDS=$(pgrep -f "rosmaster" || true)
ROSLAUNCH_PIDS=$(pgrep -f "roslaunch" || true)
ROS_NODE_PIDS=$(pgrep -f "python.*ros" || true)

# Kill processes if found
ALL_PIDS="$ROSCORE_PIDS $ROSMASTER_PIDS $ROSLAUNCH_PIDS $ROS_NODE_PIDS"
if [ -n "$ALL_PIDS" ]; then
    echo "Found existing ROS processes, cleaning up..."
    
    # Graceful termination first
    for pid in $ALL_PIDS; do
        if [ -n "$pid" ]; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    sleep 2  # Allow time for graceful shutdown
    
    # Force kill if still running
    for pid in $ALL_PIDS; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    echo "ROS process cleanup completed"
else
    echo "No existing ROS processes found"
fi
# ========================================

# ROS: Start Core with Arm Initialization
# Based on commands_to_use.md section 3
# Includes arm initialization from arm_init.sh
# Usage: ./GUI_start.sh <COBOT_MAGIC_PREFIX> [SUDO_PASSWORD]

set -e  # Exit on any error

# Check if COBOT_MAGIC_PREFIX is provided as parameter
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Error: COBOT_MAGIC_PREFIX must be provided as parameter"
    echo "Usage: $0 <COBOT_MAGIC_PREFIX> [SUDO_PASSWORD]"
    exit 1
fi

COBOT_MAGIC_PREFIX="$1"
SUDO_PASSWORD="$2"

# Check if provided path exists
if [ ! -d "$COBOT_MAGIC_PREFIX" ]; then
    echo "Error: COBOT_MAGIC_PREFIX path does not exist: $COBOT_MAGIC_PREFIX"
    exit 1
fi

echo "Starting ROS core with arm initialization..."
echo "Using COBOT_MAGIC_PREFIX: $COBOT_MAGIC_PREFIX"

# Navigate to the robotic arm directory
PIPER_DIR="$COBOT_MAGIC_PREFIX/cobot_magic/Piper_ros_private-ros-noetic"

if [ ! -d "$PIPER_DIR" ]; then
    echo "Error: Piper directory not found at $PIPER_DIR"
    echo "Please check if the path exists in your agx_src directory"
    exit 1
fi

cd "$PIPER_DIR"
echo "Changed to directory: $(pwd)"

# Activate CAN
echo "Activating CAN interfaces..."
if [ -f "can_multi_activate.sh" ]; then  # Should be `multi`, but let's keep it wrong
    # Use provided sudo password if available
    if [ -n "$SUDO_PASSWORD" ]; then
        echo "Using provided sudo password for CAN activation..."
        echo "$SUDO_PASSWORD" | sudo -S bash can_multi_activate.sh
    else
        echo "No sudo password provided, using interactive mode..."
    bash can_multi_activate.sh
    fi
    echo "CAN activation completed"
else
    echo "Warning: can_multi_activate.sh not found"
fi

# Start roscore
echo "Launching roscore..."
roscore
