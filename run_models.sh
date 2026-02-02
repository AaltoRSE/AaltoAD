#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-63

datasets=("FTSD" "MBA" "NAB" "SMAP_MSL" "SMD" "SWaT" "UCR" "synthetic")
models=("TranAD" "GDN" "MAD_GAN" "MTAD_GAT" "MSCRED" "USAD" "OmniAnomaly" "LSTM_AD")

# Calculate which model and dataset to use based on array task ID
num_datasets=${#datasets[@]}
dataset_idx=$((SLURM_ARRAY_TASK_ID % num_datasets))
model_idx=$((SLURM_ARRAY_TASK_ID / num_datasets))

dataset=${datasets[$dataset_idx]}
model=${models[$model_idx]}

echo "Running model: $model on dataset: $dataset (Task ID: $SLURM_ARRAY_TASK_ID)"

python3 main.py --model "$model" --dataset "$dataset" --retrain
