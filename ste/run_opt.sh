#!/bin/bash
#SBATCH --account=kempner_murphy_lab
#SBATCH -p kempner_requeue
#SBATCH -n 1 # Number of nodes
#SBATCH -c 8 # Number of cores
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2g
#SBATCH -t 10:00:00
#SBATCH -o OUTPUT/out_%A_%a.txt
#SBATCH -e OUTPUT/err_%A_%a.txt
#SBATCH --array=0-251
#SBATCH -J opt_policy

python opt_policy.py $SLURM_ARRAY_TASK_ID
