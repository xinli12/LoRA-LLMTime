#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=/home/xl628/courseworks/m2-cw/logs/train.log
#SBATCH --error=/home/xl628/courseworks/m2-cw/logs/train.err
#SBATCH --time=02:00:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=ampere             # GPU partition
#SBATCH --gres=gpu:4                    # Request 1 GPU
#SBATCH -A MPHIL-DIS-SL2-GPU            # Your project account

# CSD3 submit: sbatch scripts/train.sh
# CSD3 list submitted jobs: gstatement
# squeue -p ampere

# Please run this script at the ROOT of the repository ./scripts/train.sh

# cd $SLURM_SUBMIT_DIR
# module load cuda/11.4
# source .venv/bin/activate

# Create results directories
mkdir -p results/lora

# Set common variables
BATCH_SIZE=8
MAX_STEPS=1000
EVAL_STEPS=50
LORA_RANK=4
LEARNING_RATE=1e-5
CONTEXT_LENGTH=512
MODEL_PATH="results/lora/lora_r${LORA_RANK}_lr${LEARNING_RATE}_ctx${CONTEXT_LENGTH}"

# Part (a): Single LoRA configuration
echo "=== Part (a): Running single LoRA configuration with default parameters ==="
echo "Rank: $LORA_RANK, Learning Rate: $LEARNING_RATE, Context Length: $CONTEXT_LENGTH, Steps: $MAX_STEPS"
PYTHONPATH=. python src/train_lora.py \
    --rank $LORA_RANK \
    --lr $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --ctx_length $CONTEXT_LENGTH \
    --eval_steps $EVAL_STEPS \
    --output_dir results/lora
echo "=== Done! ==="
echo "Results are available in results/lora/"


# PYTHONPATH=. python src/train_lora.py \
#     --rank 4 \
#     --lr 1e-4 \
#     --max_steps 50 \
#     --batch_size 4 \
#     --ctx_length 512 \
#     --eval_steps 25 \
#     --output_dir results/lora
