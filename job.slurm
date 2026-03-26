#!/bin/bash
#SBATCH --job-name=asno_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ec9834@princeton.edu


# ── Run ───────────────────────────────────────────────────────────────────────
cd /home/ec9834/SSNO_New
export PYTHONPATH="/home/ec9834/flash-stu-2:$PYTHONPATH"
mkdir -p logs

pip install tqdm h5py numpy einops --quiet

python train.py \
    --data_glob "/home/ec9834/combined_N250_k8.h5" \
    --spatial_subsample 8 \
    --n_epochs 500 \
    --checkpoint_path best_asno_ns.pt