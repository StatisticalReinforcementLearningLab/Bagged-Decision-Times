#!/bin/bash
#SBATCH --account=murphy_lab
#SBATCH -p serial_requeue
#SBATCH -n 1
#SBATCH -c 50
#SBATCH --mem-per-cpu=1500M
#SBATCH -t 10:00:00
#SBATCH -o OUTPUT/out_%A_%a.txt
#SBATCH -e OUTPUT/err_%A_%a.txt
#SBATCH --array 0-11
#SBATCH -J exp_BSLSVI

python exp_BSLSVI.py $SLURM_ARRAY_TASK_ID
