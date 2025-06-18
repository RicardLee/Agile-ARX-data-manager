#!/bin/bash

# Lifting Column: Start Control
# Based on commands_to_use.md section 8

echo "Starting lifting column control..."

# Navigate to lifting workspace
LIFTING_DIR="$HOME/cobot_magic/lifting_ws"

if [ ! -d "$LIFTING_DIR" ]; then
    echo "Error: lifting_ws directory not found at $LIFTING_DIR"
    exit 1
fi

cd "$LIFTING_DIR"
echo "Changed to directory: $(pwd)"

# Source the development environment
if [ -f "devel/setup.bash" ]; then
    source devel/setup.bash
    echo "Sourced lifting workspace environment"
else
    echo "Error: devel/setup.bash not found"
    exit 1
fi

# Launch lifting control node
echo "Launching lifting motor control node..."
roslaunch lifting_ctrl start_850pro_motor.launch
