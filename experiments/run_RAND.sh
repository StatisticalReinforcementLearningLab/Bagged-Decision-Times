#!/bin/bash

#SBATCH --account=xxxx
#SBATCH -p xxxx
#SBATCH -n 1
#SBATCH -c 50
#SBATCH --mem-per-cpu=1g
#SBATCH -t 1:00:00
#SBATCH -o OUTPUT/out_%A_%a.txt
#SBATCH -e OUTPUT/err_%A_%a.txt
#SBATCH --array 0
#SBATCH -J exp_RAND

python exp_RAND.py $SLURM_ARRAY_TASK_ID
