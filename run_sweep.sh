#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9

echo "Worker $SLURM_ARRAY_TASK_ID starting on $(hostname)"

python3 main.py --experiment-file experiments.json --worker
