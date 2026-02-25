#!/bin/bash
#SBATCH --job-name=asno_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gm3460@princeton.edu


# ── Run ───────────────────────────────────────────────────────────────────────
cd /home/ec9834/SSNO_New

python train.py \
    --dataset 2D_DarcyFlow_beta1.0_Train.hdf5 \
    --n_traj 5000 \
    --n_epochs 200 \
    --spatial_subsample 2
