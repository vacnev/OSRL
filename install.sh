#!/bin/bash

# Install requirements
# echo "📦 Installing requirements..."
pip install -e ./DSRL
pip install -e ./OSRL
pip install -r requirements.txt

# echo "🎮 Installing DSRL MetaDrive fork..."
pip install --no-deps git+https://github.com/HenryLHH/metadrive_clean.git@main

echo "✅ Requirements and MetaDrive installed successfully"
