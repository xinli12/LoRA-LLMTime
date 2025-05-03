#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=/home/xl628/courseworks/m2-cw/logs/train_final.log
#SBATCH --error=/home/xl628/courseworks/m2-cw/logs/train_final.err
#SBATCH --time=06:00:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=ampere             # GPU partition
#SBATCH --gres=gpu:4                    # Request 1 GPU
#SBATCH -A MPHIL-DIS-SL2-GPU            # Your project account

# CSD3 submit: sbatch scripts/train_final.sh
# CSD3 list submitted jobs: gstatement
# squeue -p ampere

# Please run this script at the ROOT of the repository ./scripts/train_final.sh

# cd $SLURM_SUBMIT_DIR
# module load cuda/11.4
# source .venv/bin/activate

# Create results directories
mkdir -p results/final_model

# Set common variables
BATCH_SIZE=8
MAX_STEPS=5000
EVAL_STEPS=50
LORA_RANK=8
LEARNING_RATE=1e-4
CONTEXT_LENGTH=768
MODEL_PATH="results/final_model/lora_r${LORA_RANK}_lr${LEARNING_RATE}_ctx${CONTEXT_LENGTH}"


echo "=== Part (c): Training final model ==="
echo "Rank: $LORA_RANK, Learning Rate: $LEARNING_RATE, Context Length: $CONTEXT_LENGTH, Steps: $MAX_STEPS"
PYTHONPATH=. python src/train_lora.py \
    --rank $LORA_RANK \
    --lr $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --ctx_length $CONTEXT_LENGTH \
    --eval_steps $EVAL_STEPS \
    --output_dir results/final_model
echo "=== Done! ==="
echo "Results are available in results/final_model/"


# PYTHONPATH=. python src/train_lora.py \
#     --rank 4 \
#     --lr 1e-4 \
#     --max_steps 50 \
#     --batch_size 4 \
#     --ctx_length 512 \
#     --eval_steps 25 \
#     --output_dir results/lora
