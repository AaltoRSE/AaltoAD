#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
##SBATCH --array=0-19

echo "Worker $SLURM_ARRAY_TASK_ID starting on $(hostname)"

module load mamba
source activate TranAD

srun python3 main.py --experiment-file experiments.json --worker
