#!/usr/bin/env python3
"""
Script for training a Qwen2.5-Instruct model with LoRA adaptation on time series data.

This script implements Low-Rank Adaptation (LoRA) for the Qwen2.5-Instruct model,
applying LoRA to the query and value projection layers in the attention mechanism.
It trains the model for up to 10,000 steps and evaluates performance on the validation set.

Usage:
    python train_lora.py --rank 4 --lr 1e-5 --max_steps 10000 --batch_size 4 --ctx_length 512

Parameters tuned:
    - LoRA rank (r): Controls the parameter efficiency of the adaptation
    - Learning rate: Controls the rate of parameter updates
    - Context length: Controls the amount of context available to the model
"""

import argparse
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import torch
import wandb
from accelerate import Accelerator

sys.path.append(os.path.abspath(".."))

from qwen import load_qwen
from src.utils.flops_calculator import calculate_lora_flops

# Import project modules
from src.utils.lora import apply_lora_to_model, create_train_val_loaders, train_lora

plt.style.use("notebooks/mystyle.mplstyle")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Qwen2.5 with LoRA")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--ctx_length", type=int, default=512, help="Context length")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--output_dir", type=str, default="results/lora", help="Output directory")
    parser.add_argument("--alpha", type=int, default=None, help="LoRA alpha scaling factor (default: same as rank)")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="qwen-lora", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    return parser.parse_args()


def setup_wandb(args):
    """Initialize Weights & Biases for experiment tracking."""
    if args.use_wandb:
        run_name = args.wandb_name if args.wandb_name else f"lora_r{args.rank}_lr{args.lr}_ctx{args.ctx_length}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "rank": args.rank,
                "learning_rate": args.lr,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "context_length": args.ctx_length,
                "alpha": args.alpha if args.alpha else args.rank,
            },
        )
        logger.info(f"Initialized W&B run: {run_name}")


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


def main():
    """Main function to train and evaluate the LoRA model."""
    args = parse_args()
    setup_wandb(args)

    # Set seeds for reproducibility
    torch.manual_seed(42)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    logger.info("Loading Qwen2.5-Instruct model...")
    model, tokenizer = load_qwen()

    # Apply LoRA to the model
    logger.info(f"Applying LoRA with rank {args.rank}...")
    model = apply_lora_to_model(model, r=args.rank, target_modules=["q_proj", "v_proj"], alpha=args.alpha)

    # Document which parameters are being trained
    trainable_params = {name: p.shape for name, p in model.named_parameters() if p.requires_grad}
    logger.info("Parameters being fine-tuned:")
    for name, shape in trainable_params.items():
        logger.info(f"  {name}: {shape}")

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    train_texts, val_texts = load_data()

    # Create data loaders
    logger.info(f"Creating data loaders with context length {args.ctx_length}...")
    train_loader, val_loader = create_train_val_loaders(
        train_texts, val_texts, tokenizer, max_ctx_length=args.ctx_length, batch_size=args.batch_size
    )

    # Set up optimizer
    logger.info(f"Setting up optimizer with learning rate {args.lr}...")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    # Set up accelerator
    accelerator = Accelerator()
    logger.info(f"Accelerator device: {accelerator.device}")
    logger.info(f"Accelerator distributed type: {accelerator.distributed_type}")
    logger.info(f"Using distributed: {accelerator.use_distributed}")

    # After preparing the model, optimizer, etc. with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    logger.info(f"Model device after accelerator preparation: {model.device}")

    # You can also check a sample batch
    for batch in train_loader:
        logger.info(f"Batch device: {batch[0].device}")
        break

    # Training and evaluation
    logger.info(f"Starting LoRA training for {args.max_steps} steps...")
    run_name = f"lora_r{args.rank}_lr{args.lr}_ctx{args.ctx_length}"
    output_dir = os.path.join(args.output_dir, run_name)

    # Train the model
    results = train_lora(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        accelerator=accelerator,
        max_steps=args.max_steps,
        eval_loader=val_loader,
        eval_steps=args.eval_steps,
        logging_steps=100,
        save_steps=0,
        output_dir=output_dir,
        model_name=run_name,
        save_best=True,
        save_final=False,
    )

    # Calculate FLOPS used for training
    training_flops_info = calculate_lora_flops(
        batch_size=args.batch_size, seq_length=args.ctx_length, r=args.rank, steps=args.max_steps, is_training=True
    )

    # Calculate FLOPS used for inference/evaluation
    inference_flops_info = calculate_lora_flops(
        batch_size=args.batch_size,
        seq_length=args.ctx_length,
        r=args.rank,
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
    logger.info(
        f"Inference forward FLOPS per step: {inference_flops_info['lora_additional_forward'] + inference_flops_info['standard_forward']:,}"
    )

    # Save results to JSON instead of CSV
    results_json = {
        "learning_rate": args.lr,
        "lora_rank": args.rank,
        "context_length": args.ctx_length,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "total_flops": training_flops_info["combined_total"] + inference_flops_info["combined_total"],
        "total_training_flops": training_flops_info["combined_total"],
        "total_inference_flops": inference_flops_info["combined_total"],
        "training_flops_per_step": training_flops_info["combined_per_step"],
        "inference_flops_per_step": inference_flops_info["combined_per_step"],
        "train_losses": results["train_losses"],
        "eval_losses": results["eval_losses"],
        "eval_steps": results["eval_steps"],
        "best_eval_loss": results["best_eval_loss"],
        "steps": results["steps"],
    }

    results_json_path = os.path.join(output_dir, f"{run_name}_results.json")
    with open(results_json_path, "w") as f:
        import json

        json.dump(results_json, f, indent=2)
    logger.info(f"Results saved to {results_json_path}")

    # Plot training curve
    plt.figure(figsize=(10, 6))

    steps = list(range(1, len(results["train_losses"]) + 1))

    plt.plot(steps, results["train_losses"], label="Training Loss")
    plt.plot(results["eval_steps"], results["eval_losses"], "o-", label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"LoRA Training (r={args.rank}, lr={args.lr}, ctx={args.ctx_length})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{run_name}_training_curve.pdf")
    plt.savefig(plot_path)
    logger.info(f"Training curve saved to {plot_path}")

    # Close wandb run if active
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
