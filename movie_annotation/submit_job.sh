#!/bin/bash
#SBATCH --job-name=movie_annotate
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbnv-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module unload cudatoolkit 2>/dev/null
module load cuda/12.8

cd /u/dtyoung/movie_annotation

uv run python annotate.py
