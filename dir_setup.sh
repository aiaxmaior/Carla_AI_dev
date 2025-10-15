#!/bin/bash
# Q-DRIVE Cortex Directory Structure Setup Script

echo "🚀 Setting up Q-DRIVE Cortex modular directory structure..."

# Create the main utility directories
mkdir -p Utility/Monitor
mkdir -p Utility/Hardware
mkdir -p Utility/Audio
mkdir -p Utility/UI
mkdir -p Utility/Data
mkdir -p Core/Sensors
mkdir -p Core/Vision
mkdir -p Core/Controls
mkdir -p Core/Simulation

# Create __init__.py files to make directories proper Python packages
touch Utility/__init__.py
touch Utility/Monitor/__init__.py
touch Utility/Hardware/__init__.py
touch Utility/Audio/__init__.py
touch Utility/UI/__init__.py
touch Utility/Data/__init__.py
touch Core/__init__.py
touch Core/Sensors/__init__.py
touch Core/Vision/__init__.py
touch Core/Controls/__init__.py
touch Core/Simulation/__init__.py

echo "✅ Directory structure created!"
echo ""
echo "📁 Project structure:"
echo "├── Core/"
echo "│   ├── __init__.py"
echo "│   ├── Sensors/"
echo "│   │   └── __init__.py"
echo "│   ├── Vision/"
echo "│   │   └── __init__.py"
echo "│   ├── Controls/"
echo "│   │   └── __init__.py"
echo "│   └── Simulation/"
echo "│       └── __init__.py"
echo "├── Utility/"
echo "│   ├── __init__.py"
echo "│   ├── Monitor/"
echo "│   │   └── __init__.py"
echo "│   ├── Hardware/"
echo "│   │   └── __init__.py"
echo "│   ├── Audio/"
echo "│   │   └── __init__.py"
echo "│   ├── UI/"
echo "│   │   └── __init__.py"
echo "│   └── Data/"
echo "│       └── __init__.py"
echo "├── Main.py"
echo "├── HUD.py"
echo "├── World.py"
echo "└── ..."
echo ""
echo "🎯 Next steps:"
echo "1. Move your files to appropriate directories"
echo "2. Update import statements in your code"
echo "3. Test imports with: python -c 'from Utility.Monitor.DynamicMonitor import DynamicMonitor'"