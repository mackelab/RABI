#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --partition=cpu-short
#SBATCH --mem=20000
#SBATCH --output=/mnt/qb/work/macke/mdeistler57/Documents/pyloric_sims/results/setup1/outfiles/out_%j.out
#SBATCH --error=/mnt/qb/work/macke/mdeistler57/Documents/pyloric_sims/results/setup1/outfiles/out_%j.err
#SBATCH --time=0-10:00

python simulate.py
