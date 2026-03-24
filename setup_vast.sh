#!/bin/bash
# Setup script for running SSNO_New on a Vast.ai instance.
# Run once after cloning the repo:  bash setup_vast.sh

set -e

echo "=== Installing Python dependencies ==="
pip install torch h5py numpy

echo ""
echo "=== Cloning flash-stu-2 (MiniSTU) ==="
if [ ! -d "/workspace/flash-stu-2" ]; then
    git clone https://github.com/hazan-lab/flash-stu-2.git /workspace/flash-stu-2
else
    echo "flash-stu-2 already exists, skipping clone."
fi

echo ""
echo "=== Setting PYTHONPATH ==="
export PYTHONPATH="/workspace/flash-stu-2:$PYTHONPATH"

# Make it permanent
if ! grep -q "flash-stu-2" ~/.bashrc; then
    echo 'export PYTHONPATH="/workspace/flash-stu-2:$PYTHONPATH"' >> ~/.bashrc
    echo "Added to ~/.bashrc"
else
    echo "PYTHONPATH already set in ~/.bashrc"
fi

echo ""
echo "=== Creating directories ==="
mkdir -p /root/data logs

echo ""
echo "=== Done! ==="
echo "To start training, run:"
echo ""
echo "  PYTORCH_ALLOC_CONF=expandable_segments:True python train.py \\"
echo "      --data_glob '/root/data/combined_N250_k8.h5' \\"
echo "      --spatial_subsample 8 \\"
echo "      --n_epochs 500 \\"
echo "      --checkpoint_path best_asno_ns.pt"
