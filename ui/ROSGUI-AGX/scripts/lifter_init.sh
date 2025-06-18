#!/bin/bash

# Lifting Column: Initialize
# Based on commands_to_use.md section 8

echo "Initializing lifting column..."

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

# Check if lifting control service is available
echo "Checking if lifting motor service is available..."
rosservice list | grep -q "/lifter_1/LiftingMotorService"

if [ $? -ne 0 ]; then
    echo "Error: Lifting motor service not found"
    echo "Please make sure the lifting control node is running"
    exit 1
fi

# Initialize the lifting motor
echo "Initializing lifting motor (val: 0, mode: 1)..."
rosservice call /lifter_1/LiftingMotorService "{val: 0, mode: 1}"

echo "Lifting column initialization completed"
