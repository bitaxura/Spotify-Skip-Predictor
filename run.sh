#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "Spotify Skip Prediction & Audio Behavior Modeling Pipeline"
echo "=========================================================="

# Check if venv python exists
if [ -f "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "Using Python: $PYTHON_BIN"

# Execute orchestrator
$PYTHON_BIN main.py --step all

echo "Done! Review results in 'results/figures/' and 'results/tables/'."
