#!/bin/bash
# ==============================================================================
# PAPO Execution Script for University RUSA Server (80GB RAM Environment)
# Bypasses Ray Cluster Crashes using Standalone Scaled-Down Pipeline
# ==============================================================================

set -e

echo "=== PAPO Research Standalone Pipeline (RUSA Server Adaptation) ==="
echo "Checking environment..."

# 1. Activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment for RUSA..."
    python3 -m venv venv
    source venv/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets peft accelerate matplotlib pandas
fi

# 2. Check GPU
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 3. Run Standalone Extended PAPO Training (Can scale up to 500+ steps on RUSA)
echo "Starting PAPO Training on RUSA..."
python3 train_papo_extended.py

# 4. Run Evaluation
echo "Running Validation Evaluation on MMK12..."
python3 evaluate_comparison.py

echo "=== Finished! Results saved to dual_experiment_comparison.json and papo_extended_loss_curves.png ==="

