#!/bin/bash
#SBATCH --job-name=abcd-svm-sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=1-50%10
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=general

module load anaconda/2023.03
source activate abcd

wandb agent <SWEEP_ID>
