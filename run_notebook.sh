#!/bin/bash
set -e

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install jupyter notebook numpy pandas matplotlib seaborn Pillow scikit-learn tensorflow

# Execute the notebook
echo "Executing notebook..."
jupyter nbconvert --to notebook --execute Untitled.ipynb --output Untitled_executed.ipynb

echo "Done."
