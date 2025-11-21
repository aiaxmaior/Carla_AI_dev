#!/bin/bash
# This script launches the CARLA Driving Performance System.
# It assumes:
# 1. You have activated a conda environment with required packages
# 2. CARLA_ROOT environment variable is set, OR you provide it as an argument
#    OR you modify the CARLA_ROOT_PATH variable below.

# --- Configuration Variables ---
# Path to your CARLA installation directory (e.g., /opt/carla-simulator)
# IMPORTANT: Replace this with the actual path if CARLA_ROOT is not an environment variable
# or if you prefer to hardcode it.
CARLA_ROOT_PATH="${CARLA_ROOT:-./}"

# Path to the directory where Main.py and other Python scripts are located.
# This assumes your scripts are in the script's directory.
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # Gets script's directory

# --- Detect Python from conda environment ---
# Check if we're in a conda environment
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
    echo "✓ Using Python from conda environment: $CONDA_DEFAULT_ENV"
    echo "  Python path: $PYTHON_CMD"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "⚠ Using system Python (no conda environment detected)"
    echo "  Python path: $(which python)"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "⚠ Using system Python3 (no conda environment detected)"
    echo "  Python path: $(which python3)"
else
    echo "❌ Error: No Python interpreter found!"
    echo "Please activate your conda environment or ensure Python is installed."
    exit 1
fi

# Verify Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "  $PYTHON_VERSION"

# --- Script Logic ---
echo ""
echo "============================================"
echo "Starting QDrive Safety Simulator"
echo "============================================"
echo "Author: Arjun Joshi, Joshi.Arjun.K@Gmail.com"
echo "https://github.com/aiaxmaior/Carla_AI_dev"
echo ""
echo "CARLA Root Path: $CARLA_ROOT_PATH"
echo "Project Directory: $PROJECT_DIR"
echo "============================================"
echo ""

# Check if CARLA_ROOT_PATH exists
if [ ! -d "$CARLA_ROOT_PATH" ]; then
    echo "❌ Error: CARLA_ROOT_PATH '$CARLA_ROOT_PATH' does not exist or is not a directory."
    echo "Please edit this script and set CARLA_ROOT_PATH to your CARLA installation directory."
    exit 1
fi

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "❌ Error: Could not change to project directory '$PROJECT_DIR'. Exiting."; exit 1; }

# Check if Main.py exists
if [ ! -f "Main.py" ]; then
    echo "❌ Error: Main.py not found in $PROJECT_DIR"
    echo "Please ensure you're running this script from the correct directory."
    exit 1
fi

# --- Execute the Python script with arguments ---
# The "$@" will expand to all command-line arguments passed to this shell script.
# This allows you to pass arguments like --num-vehicles or --host directly when running this .sh file.

echo "Launching simulator..."
echo ""

$PYTHON_CMD Main.py \
    --host localhost \
    --port 2000 \
    --num-vehicles 10 \
    --num-pedestrians 10 \
    --carla-root "$CARLA_ROOT_PATH" \
    --display 1 \
    "$@" # This line passes all arguments from the shell script to the Python script

# Check the exit status of the python command
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ QDrive Driver Safety System exited successfully."
else
    echo "❌ QDrive Driver Safety System exited with error code $EXIT_CODE."
    echo "Please check the logs above for details."
fi

echo ""
echo "============================================"
echo "Script finished."
echo "============================================"

exit $EXIT_CODE
