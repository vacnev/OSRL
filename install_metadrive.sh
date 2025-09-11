#!/bin/bash
# Install MetaDrive fork with dependencies

echo "🎮 Installing MetaDrive dependencies..."
pip install panda3d==1.10.10

echo "🎮 Installing DSRL MetaDrive fork..."
pip install --no-deps git+https://github.com/HenryLHH/metadrive_clean.git@main

echo "✅ MetaDrive fork installed successfully"
