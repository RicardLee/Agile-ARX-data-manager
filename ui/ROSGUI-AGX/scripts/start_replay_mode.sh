#!/bin/bash

# ========== Fail-safe Design ==========
# Kill any existing piper roslaunch processes to prevent conflicts
echo "Checking for existing piper roslaunch processes..."

# Find and kill processes with command starting with "roslaunch piper start_ms_piper.launch"
PIPER_PIDS=$(pgrep -f "roslaunch piper start_ms_piper.launch" || true)

if [ -n "$PIPER_PIDS" ]; then
    echo "Found existing piper roslaunch processes, cleaning up..."
    
    # Graceful termination first
    for pid in $PIPER_PIDS; do
        if [ -n "$pid" ]; then
            echo "Terminating piper process PID: $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    sleep 3  # Give roslaunch more time to clean up its child processes
    
    # Force kill if still running
    for pid in $PIPER_PIDS; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Force killing piper process PID: $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    echo "Piper process cleanup completed"
else
    echo "No existing piper roslaunch processes found"
fi
# ========================================

# Start Replay Mode Script  
# This script starts the ROS system in replay mode (mode:=1) for data replay
# Usage: ./start_replay_mode.sh <COBOT_MAGIC_PREFIX>

set -e  # Exit on any error

# Check if COBOT_MAGIC_PREFIX is provided as parameter
if [ $# -ne 1 ]; then
    echo "Error: COBOT_MAGIC_PREFIX must be provided as parameter"
    echo "Usage: $0 <COBOT_MAGIC_PREFIX>"
    exit 1
fi

COBOT_MAGIC_PREFIX="$1"

# Check if provided path exists
if [ ! -d "$COBOT_MAGIC_PREFIX" ]; then
    echo "Error: COBOT_MAGIC_PREFIX path does not exist: $COBOT_MAGIC_PREFIX"
    exit 1
fi

# Define paths
ROS_WS_PATH="$COBOT_MAGIC_PREFIX/cobot_magic/Piper_ros_private-ros-noetic"
SETUP_SCRIPT="$ROS_WS_PATH/devel/setup.bash"

# Check if setup script exists
if [ ! -f "$SETUP_SCRIPT" ]; then
    echo "Error: ROS setup script not found at $SETUP_SCRIPT"
    exit 1
fi

echo "Starting Collection Mode..."
echo "Setup script: $SETUP_SCRIPT"

# Source ROS setup and launch collection mode
source "$SETUP_SCRIPT"
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

echo "Replay mode started successfully" 