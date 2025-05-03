#!/bin/bash
#SBATCH --job-name=evaluate_baseline
#SBATCH --output=/home/xl628/courseworks/m2-cw/logs/evaluate_baseline.log
#SBATCH --error=/home/xl628/courseworks/m2-cw/logs/evaluate_baseline.err
#SBATCH --time=02:00:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=ampere             # GPU partition
#SBATCH --gres=gpu:4                    # Request 1 GPU
#SBATCH -A MPHIL-DIS-SL2-GPU            # Your project account

# CSD3 submit: sbatch scripts/evaluate_baseline.sh
# CSD3 list submitted jobs: gstatement
# squeue -p ampere

# Please run this script at the ROOT of the repository ./scripts/evaluate_baseline.sh

# cd $SLURM_SUBMIT_DIR
# module load cuda/11.4
# source .venv/bin/activate

# Create results directories
mkdir -p results/baseline_evaluation

# Set common variables
NUM_SAMPLES=-1  # Set to -1 to use all test samples
FORECAST_LENGTH=5
CONTEXT_LENGTH=20
UNCERTAINTY_SAMPLES=20
BATCH_SIZE=32

echo "=== Baseline Evaluation of Qwen2.5-Instruct for Time Series Forecasting ==="
echo "This will evaluate the untrained model's performance on time series forecasting"
echo "Number of samples: $NUM_SAMPLES"
echo "Forecast length: $FORECAST_LENGTH"
echo "Context length: $CONTEXT_LENGTH"
echo "Uncertainty samples: $UNCERTAINTY_SAMPLES"
echo "Batch size: $BATCH_SIZE"

# Run evaluation
PYTHONPATH=. python src/evaluate_baseline.py \
    --num_samples $NUM_SAMPLES \
    --forecast_length $FORECAST_LENGTH \
    --context_length $CONTEXT_LENGTH \
    --uncertainty_samples $UNCERTAINTY_SAMPLES \
    --output_dir results/baseline_evaluation \
    --batch_size $BATCH_SIZE \
    --data_path data/formatted_test.pkl

echo "=== Done! ==="
echo "Results are available in results/baseline_evaluation/"
