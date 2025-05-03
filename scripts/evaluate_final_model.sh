#!/bin/bash
#SBATCH --job-name=evaluate_final_model
#SBATCH --output=/home/xl628/courseworks/m2-cw/logs/evaluate_final_model.log
#SBATCH --error=/home/xl628/courseworks/m2-cw/logs/evaluate_final_model.err
#SBATCH --time=04:00:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=ampere             # GPU partition
#SBATCH --gres=gpu:4                    # Request 1 GPU
#SBATCH -A MPHIL-DIS-SL2-GPU            # Your project account

# CSD3 submit: sbatch scripts/evaluate_final_model.sh
# CSD3 list submitted jobs: gstatement
# squeue -p ampere

# Please run this script at the ROOT of the repository ./scripts/evaluate_final_model.sh

# cd $SLURM_SUBMIT_DIR
# module load cuda/11.4
# source .venv/bin/activate

# Create results directories
mkdir -p results/final_model_evaluation

# Set common variables
NUM_SAMPLES=-1  # Set to -1 to use all test samples
FORECAST_LENGTH=5
CONTEXT_LENGTH=20
UNCERTAINTY_SAMPLES=20
BATCH_SIZE=32
LORA_RANK=8
OUTPUT_DIR="results/final_model_evaluation"
DATA_PATH="data/formatted_test.pkl"
MODEL_PATH="results/final_model/lora_r8_lr0.0001_ctx768/lora_r8_lr0.0001_ctx768_best"
TOKENIZER_NAME="Qwen/Qwen2.5-0.5B-Instruct"

echo "=== Custom Model Evaluation for Time Series Forecasting ==="
echo "This will evaluate the custom model's performance on time series forecasting"
echo "Model path: $MODEL_PATH"
echo "Tokenizer: $TOKENIZER_NAME"
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
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --tokenizer_name $TOKENIZER_NAME \
    --lora_rank $LORA_RANK

echo "=== Done! ==="
echo "Results are available in results/final_model_evaluation/"
