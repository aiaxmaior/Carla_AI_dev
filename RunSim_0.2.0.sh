#!/bin/bash

# This script launches the CARLA Driving Performance System.
# It assumes:
# 1. Python 3 is installed and accessible via 'python3'.
# 2. CARLA_ROOT environment variable is set, OR you provide it as an argument
#    OR you modify the CARLA_ROOT_PATH variable below.

# --- Configuration Variables ---

# Path to your CARLA installation directory (e.g., /opt/carla-simulator)
# IMPORTANT: Replace this with the actual path if CARLA_ROOT is not an environment variable
# or if you prefer to hardcode it.
CARLA_ROOT_PATH="${CARLA_ROOT:-./}"

# Path to the directory where Main.py and other Python scripts are located.
# This assumes your scripts are in a directory named 'carla_project' within your home directory.
# Adjust this path as necessary.
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # Gets script's directory

# --- Script Logic ---

echo "Starting CARLA Driving Performance System..."
echo "CARLA Root Path: $CARLA_ROOT_PATH"
echo "Project Directory: $PROJECT_DIR"

# Check if CARLA_ROOT_PATH exists
if [ ! -d "$CARLA_ROOT_PATH" ]; then
    echo "Error: CARLA_ROOT_PATH '$CARLA_ROOT_PATH' does not exist or is not a directory."
    echo "Please edit this script and set CARLA_ROOT_PATH to your CARLA installation directory."
    exit 1
fi

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Error: Could not change to project directory '$PROJECT_DIR'. Exiting."; exit 1; }

# --- Execute the Python script with arguments ---
# The "$@" will expand to all command-line arguments passed to this shell script.
# This allows you to pass arguments like --num-vehicles or --host directly when running this .sh file.

python3 Main.py \
    --host localhost \
    --port 2000 \
    --num-vehicles 20 \
    --num-pedestrians 20 \
    --carla-root "$CARLA_ROOT_PATH" \
    "$@" # This line passes all arguments from the shell script to the Python script

# Check the exit status of the python command
if [ $? -eq 0 ]; then
    echo "CARLA Driving Performance System exited successfully."
else
    echo "CARLA Driving Performance System exited with an error. Please check the logs above."
fi

echo "Script finished."

