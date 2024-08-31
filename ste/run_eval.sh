#!/bin/bash
#SBATCH --account=kempner_murphy_lab
#SBATCH -p kempner_requeue
#SBATCH -n 1 # Number of nodes
#SBATCH -c 1 # Number of cores
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1g
#SBATCH -t 20:00:00 # Runtime in D-HH:MM
#SBATCH -o OUTPUT/out_%A_%a.txt
#SBATCH -e OUTPUT/err_%A_%a.txt
#SBATCH --array=0-251
#SBATCH --dependency=afterok:44105917 # after opt_policy is finished
#SBATCH -J eval_ste

python eval_ste.py $SLURM_ARRAY_TASK_ID
