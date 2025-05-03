#!/usr/bin/env python3
"""
Script for LoRA hyperparameter search on the Qwen2.5-Instruct model.

This script conducts a systematic search over LoRA hyperparameters for the Qwen2.5-Instruct model:
1. Learning rates: 1e-5, 5e-5, 1e-4
2. LoRA ranks: 2, 4, 8
3. Context lengths: 128, 512, 768 (for the best hyperparameter configuration found)

All experiments are tracked with metrics, loss curves, and FLOPS calculations.

Usage:
    python hyperparameter_search.py --output_dir results/hyperparameter_search [--use_wandb]
"""

import argparse
import itertools
import logging
import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from accelerate import Accelerator

sys.path.append(os.path.abspath(".."))

from qwen import load_qwen
from src.utils.flops_calculator import calculate_lora_flops

# Import project modules
from src.utils.lora import apply_lora_to_model, create_train_val_loaders, train_lora

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("notebooks/mystyle.mplstyle")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LoRA Hyperparameter Search")
    parser.add_argument("--output_dir", type=str, default="results/hyperparameter_search", help="Directory to save results")
    parser.add_argument("--max_steps_main", type=int, default=10000, help="Maximum steps for main hyperparameter search")
    parser.add_argument("--max_steps_context", type=int, default=2000, help="Maximum steps for context length experiments")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="qwen-lora-hyperparam", help="W&B project name")
    return parser.parse_args()


def setup_wandb(args, experiment_name):
    """Initialize Weights & Biases for experiment tracking."""
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=experiment_name,
            config={
                "max_steps_main": args.max_steps_main,
                "max_steps_context": args.max_steps_context,
                "batch_size": args.batch_size,
                "eval_steps": args.eval_steps,
            },
            group="hyperparameter_search",
        )
        logger.info(f"Initialized W&B run: {experiment_name}")
        return wandb.run.id
    return None


def load_data():
    """Load and preprocess the training and validation data."""
    data_path = "data/formatted_train.pkl"
    with open(data_path, "rb") as f:
        formatted_train = pickle.load(f)

    # Split into train/validation (80/20 split)
    train_size = int(0.8 * len(formatted_train))
    train_texts = formatted_train[:train_size]
    val_texts = formatted_train[train_size:]

    logger.info(f"Loaded {len(train_texts)} training sequences and {len(val_texts)} validation sequences")
    return train_texts, val_texts


def run_experiment(model, tokenizer, train_texts, val_texts, args, learning_rate, lora_rank, context_length, max_steps, experiment_name):
    """Run a single experiment with specific hyperparameters."""
    logger.info(f"Starting experiment: {experiment_name}")

    # Reset and apply LoRA to a fresh model
    model, tokenizer = load_qwen()
    model = apply_lora_to_model(model, r=lora_rank, target_modules=["q_proj", "v_proj"])

    # Create data loaders
    train_loader, val_loader = create_train_val_loaders(
        train_texts, val_texts, tokenizer, max_ctx_length=context_length, batch_size=args.batch_size
    )

    # Set up optimizer
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)

    # Set up accelerator
    accelerator = Accelerator()

    # Log device information
    logger.info(f"Accelerator device: {accelerator.device}")
    logger.info(f"Accelerator distributed type: {accelerator.distributed_type}")
    logger.info(f"Using distributed: {accelerator.use_distributed}")

    # Set up output directory
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize W&B if enabled
    wandb_run_id = None
    if args.use_wandb:
        wandb_run_id = setup_wandb(args, experiment_name)

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    logger.info(f"Model device after accelerator preparation: {model.device}")

    # Check a sample batch to verify device
    for batch in train_loader:
        logger.info(f"Batch device: {batch[0].device}")
        break

    # Train the model
    results = train_lora(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        accelerator=accelerator,
        max_steps=max_steps,
        eval_loader=val_loader,
        eval_steps=args.eval_steps,
        logging_steps=100,
        save_steps=0,
        output_dir=output_dir,
        model_name=experiment_name,
        save_best=True,
        save_final=False,  # Don't save final model to save disk space
    )

    # Calculate FLOPS used for training
    training_flops_info = calculate_lora_flops(
        batch_size=args.batch_size, seq_length=context_length, r=lora_rank, steps=max_steps, is_training=True
    )

    # Calculate FLOPS used for inference/evaluation
    inference_flops_info = calculate_lora_flops(
        batch_size=args.batch_size,
        seq_length=context_length,
        r=lora_rank,
        steps=len(results["eval_steps"]),  # Number of evaluation runs
        is_training=False,
    )

    # Log FLOPS information
    logger.info(f"Total FLOPS: {training_flops_info['combined_total'] + inference_flops_info['combined_total']:,}")
    logger.info(f"Total training FLOPS: {training_flops_info['combined_total']:,}")
    logger.info(
        f"Training forward FLOPS per step: {training_flops_info['lora_additional_forward'] + training_flops_info['standard_forward']:,}"
    )
    logger.info(
        f"Training backward FLOPS per step: {training_flops_info['lora_additional_backward'] + training_flops_info['standard_backward']:,}"
    )
    logger.info(f"Total inference FLOPS: {inference_flops_info['combined_total']:,}")

    # Get best and final eval loss
    best_eval_loss = min(results["eval_losses"]) if results["eval_losses"] else float("inf")
    final_eval_loss = results["eval_losses"][-1] if results["eval_losses"] else float("inf")

    # Store experiment details
    experiment_data = {
        "name": experiment_name,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "context_length": context_length,
        "max_steps": max_steps,
        "batch_size": args.batch_size,
        "total_flops": training_flops_info["combined_total"] + inference_flops_info["combined_total"],
        "total_training_flops": training_flops_info["combined_total"],
        "total_inference_flops": inference_flops_info["combined_total"],
        "training_flops_per_step": training_flops_info["combined_per_step"],
        "inference_flops_per_step": inference_flops_info["combined_per_step"],
        "train_losses": results["train_losses"],
        "eval_losses": results["eval_losses"],
        "eval_steps": results["eval_steps"],
        "best_eval_loss": best_eval_loss,
        "final_eval_loss": final_eval_loss,
        "wandb_run_id": wandb_run_id,
    }

    results_json_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(results_json_path, "w") as f:
        import json

        json.dump(experiment_data, f, indent=2)
    logger.info(f"Results saved to {results_json_path}")

    # Create summary row for this experiment
    summary_row = {
        "name": experiment_name,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "context_length": context_length,
        "best_eval_loss": best_eval_loss,
        "final_eval_loss": final_eval_loss,
        "total_flops": training_flops_info["combined_total"] + inference_flops_info["combined_total"],
    }

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(results["train_losses"]) + 1), results["train_losses"], label="Training Loss")
    plt.plot(results["eval_steps"], results["eval_losses"], "o-", label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"LoRA Training (r={lora_rank}, lr={learning_rate}, ctx={context_length})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_training_curve.pdf"))
    plt.close()

    # Close W&B run if active
    if args.use_wandb:
        wandb.finish()

    # Return a summary row for analysis
    return summary_row


def run_lr_rank_search(args, train_texts, val_texts, model, tokenizer):
    """Run hyperparameter search for learning rate and LoRA rank."""
    # Define hyperparameter grid
    learning_rates = [1e-5, 5e-5, 1e-4]
    lora_ranks = [2, 4, 8]

    # Main hyperparameter search
    logger.info("Starting main hyperparameter search")
    logger.info(f"Learning rates: {learning_rates}")
    logger.info(f"LoRA ranks: {lora_ranks}")
    logger.info(f"Max steps: {args.max_steps_main}")

    # Run experiments for all combinations of learning rate and rank
    results = []
    for lr, rank in itertools.product(learning_rates, lora_ranks):
        experiment_name = f"lora_r{rank}_lr{lr}_ctx512"

        experiment_result = run_experiment(
            model=model,
            tokenizer=tokenizer,
            train_texts=train_texts,
            val_texts=val_texts,
            args=args,
            learning_rate=lr,
            lora_rank=rank,
            context_length=512,  # Fixed for main search
            max_steps=args.max_steps_main,
            experiment_name=experiment_name,
        )

        results.append(experiment_result)

    # Save all results as a single CSV
    results_df = pd.DataFrame(results)
    all_results_path = os.path.join(args.output_dir, "all_results.csv")
    results_df.to_csv(all_results_path, index=False)
    logger.info(f"All results saved to {all_results_path}")

    # Find the best hyperparameters based on validation loss
    best_result = min(results, key=lambda x: x["best_eval_loss"])
    best_lr = best_result["learning_rate"]
    best_rank = best_result["lora_rank"]

    logger.info("Best hyperparameters found:")
    logger.info(f"  Learning rate: {best_lr}")
    logger.info(f"  LoRA rank: {best_rank}")
    logger.info(f"  Best validation loss: {best_result['best_eval_loss']:.4f}")

    # Create and save summary heatmap
    plt.figure(figsize=(10, 8))
    pivot_table = results_df.pivot_table(index="lora_rank", columns="learning_rate", values="best_eval_loss")
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis_r")
    plt.title("Validation Loss by LoRA Rank and Learning Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "hyperparam_heatmap.pdf"))
    plt.close()

    # FLOPS usage heatmap
    plt.figure(figsize=(10, 8))
    flops_pivot = results_df.pivot_table(index="lora_rank", columns="learning_rate", values="total_flops")
    sns.heatmap(flops_pivot / 1e12, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Total FLOPS (in Teraflops) by Configuration")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "flops_heatmap.pdf"))
    plt.close()

    return results, best_lr, best_rank, best_result


def run_context_length_search(args, train_texts, val_texts, model, tokenizer, best_lr, best_rank, best_result, results):
    """Run experiments with different context lengths using the best hyperparameters."""
    logger.info("Starting context length experiments with best hyperparameters")
    context_lengths = [128, 512, 768]
    context_results = []

    for ctx_len in context_lengths:
        experiment_name = f"lora_r{best_rank}_lr{best_lr}_ctx{ctx_len}"

        # Skip if we already did this configuration in the main search
        if ctx_len == 512:
            # Reuse the result from the main search
            matching_result = next(
                r for r in results if r["lora_rank"] == best_rank and r["learning_rate"] == best_lr and r["context_length"] == 512
            )
            context_results.append(matching_result)
            continue

        experiment_result = run_experiment(
            model=model,
            tokenizer=tokenizer,
            train_texts=train_texts,
            val_texts=val_texts,
            args=args,
            learning_rate=best_lr,
            lora_rank=best_rank,
            context_length=ctx_len,
            max_steps=args.max_steps_context,  # Shorter runs for context experiments
            experiment_name=experiment_name,
        )

        context_results.append(experiment_result)

    # Save context experiment results
    context_df = pd.DataFrame(context_results)
    context_df.to_csv(os.path.join(args.output_dir, "context_results.csv"), index=False)

    # Plot context length vs. loss
    plt.figure(figsize=(10, 6))
    plt.plot(context_df["context_length"], context_df["best_eval_loss"], "o-")
    plt.xlabel("Context Length")
    plt.ylabel("Best Validation Loss")
    plt.title(f"Effect of Context Length (LoRA rank={best_rank}, lr={best_lr})")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "context_length_effect.pdf"))
    plt.close()

    return context_results


def main():
    """Run the hyperparameter search."""
    args = parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Log CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    train_texts, val_texts = load_data()

    # Load model and tokenizer (initial load for examination)
    model, tokenizer = load_qwen()

    # Run learning rate and LoRA rank search
    results, best_lr, best_rank, best_result = run_lr_rank_search(args, train_texts, val_texts, model, tokenizer)

    # Run context length search with the best parameters
    context_results = run_context_length_search(args, train_texts, val_texts, model, tokenizer, best_lr, best_rank, best_result, results)

    # Create FLOPS summary table
    flops_data = []
    for result in results + context_results:
        # Avoid duplicates by checking if experiment is already in the list
        if not any(d["Experiment"] == result["name"] for d in flops_data):
            flops_data.append(
                {
                    "Experiment": result["name"],
                    "Learning Rate": result["learning_rate"],
                    "LoRA Rank": result["lora_rank"],
                    "Context Length": result["context_length"],
                    "Steps": args.max_steps_main if result["context_length"] == 512 else args.max_steps_context,
                    "Total FLOPS (T)": result["total_flops"] / 1e12,
                    "Best Loss": result["best_eval_loss"],
                }
            )

    flops_df = pd.DataFrame(flops_data)
    flops_df.to_csv(os.path.join(args.output_dir, "flops_summary.csv"), index=False)

    # Generate markdown summary report
    report = f"""# LoRA Hyperparameter Search Results

## Best Configuration
- Learning Rate: {best_lr}
- LoRA Rank: {best_rank}
- Best Validation Loss: {best_result['best_eval_loss']:.4f}

## FLOPS Usage Summary
{flops_df.to_markdown(index=False)}

## Context Length Effect
The best configuration was tested with different context lengths ({', '.join(map(str, [128, 512, 768]))}).
See `context_length_effect.pdf` for visualization.

## Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(os.path.join(args.output_dir, "summary_report.md"), "w") as f:
        f.write(report)

    logger.info(f"Hyperparameter search completed. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
