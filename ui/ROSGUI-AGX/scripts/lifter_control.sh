#!/bin/bash

# Lifting Column: Parameterized Control
# Usage: ./lifter_control.sh <val> <mode>
# val: position value (range: -800 to 0) - ignored when mode=1
# mode: 0 (position control) or 1 (initialize - val forced to 0)

# Parse arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <val> <mode>"
    echo "  val:  position value (-800 to 0, ignored when mode=1)"
    echo "  mode: 0 (position control) or 1 (initialize, val forced to 0)"
    exit 1
fi

VAL=$1
MODE=$2

# Validate arguments
if [ "$MODE" != "0" ] && [ "$MODE" != "1" ]; then
    echo "Error: mode must be 0 or 1"
    exit 1
fi

if [ "$VAL" -lt -800 ] || [ "$VAL" -gt 0 ]; then
    echo "Error: val must be between -800 and 0"
    exit 1
fi

echo "Controlling lifting column (val: $VAL, mode: $MODE)..."

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

# Execute the lifting command
if [ "$MODE" = "1" ]; then
    echo "Initializing lifting motor..."
    # For initialization, val must always be 0 (hardcoded for safety)
    rosservice call /lifter_1/LiftingMotorService "{val: 0, mode: 1}"
else
    echo "Moving lifting motor to position $VAL..."
    rosservice call /lifter_1/LiftingMotorService "{val: $VAL, mode: 0}"
fi

if [ $? -eq 0 ]; then
    echo "Command executed successfully"
    if [ "$MODE" = "0" ]; then
        echo "You can monitor the status with:"
        echo "rostopic echo /lifter_1/LiftMotorStatePub"
    fi
else
    echo "Error: Command execution failed"
    exit 1
fi 