#!/usr/bin/env bash
set -e

# Name of virtual environment folder
VENV_DIR="venv"

# Create virtual environment if not exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Running main.py..."
python main.py

echo "Deactivating virtual environment..."
deactivate
