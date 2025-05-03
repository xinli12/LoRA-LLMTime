#!/bin/bash
#SBATCH --job-name=hyperparam_search
#SBATCH --output=/home/xl628/courseworks/m2-cw/logs/hyperparam_search.log
#SBATCH --error=/home/xl628/courseworks/m2-cw/logs/hyperparam_search.err
#SBATCH --time=12:00:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=ampere             # GPU partition
#SBATCH --gres=gpu:4                    # Request 1 GPU
#SBATCH -A MPHIL-DIS-SL2-GPU            # Your project account

# CSD3 submit: sbatch scripts/hyperparam_search.sh
# CSD3 list submitted jobs: gstatement
# squeue -p ampere

# Please run this script at the ROOT of the repository ./scripts/hyperparam_search.sh

# cd $SLURM_SUBMIT_DIR
# module load cuda/11.4
# source .venv/bin/activate

# Set common variables
BATCH_SIZE=8
MAX_STEPS=250
MAX_STEPS_CONTEXT=200
EVAL_STEPS=25

# Part (b): Hyperparameter search
echo "=== Part (b): Running hyperparameter search ==="
echo "This will test combinations of:"
echo "- Learning rates: 1e-5, 5e-5, 1e-4"
echo "- LoRA ranks: 2, 4, 8"
echo "- Context lengths: 128, 512, 768 (for best configuration)"
echo


PYTHONPATH=. python src/hyperparameter_search.py \
    --output_dir results/hyperparameter_search \
    --max_steps_main $MAX_STEPS \
    --max_steps_context $MAX_STEPS_CONTEXT \
    --batch_size $BATCH_SIZE \
    --eval_steps $EVAL_STEPS

echo
echo "=== Hyperparameter search completed ==="
echo "Results are available in results/hyperparameter_search/"
echo "See summary_report.md for an overview of the results."

echo
echo "=== All experiments completed ==="
echo "The results are available in the results/ directory."


# !PYTHONPATH=. python src/hyperparameter_search.py \
#       --output_dir results/hyperparameter_search \
#       --max_steps_main 50 \
#       --max_steps_context 50 \
#       --batch_size 8 \
#       --eval_steps 25
