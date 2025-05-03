#!/usr/bin/env python3
"""
Script for evaluating time series forecasting models.

This script provides a comprehensive evaluation framework for assessing time series
forecasting performance of language models. It supports:

1. Evaluating both baseline (untrained) and fine-tuned LoRA models
2. Generating forecasts with uncertainty bounds via Monte Carlo sampling
3. Comparing against naive persistence-based forecasts
4. Computing standard error metrics (RMSE, MAE, MAPE) and uncertainty metrics (coverage, interval width)
5. Visualizing forecasts with uncertainty bounds
"""

import argparse
import json
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer

from qwen import load_qwen

# Import custom modules
from src.preprocessor import LLMTIMEPreprocessor
from src.utils.evaluator import (
    evaluate_batch_forecasts,
    evaluate_forecasts_with_uncertainty,
    visualize_multiple_forecasts_with_uncertainty,
)
from src.utils.generator import (
    batch_generate_forecasts_with_uncertainty,
    naive_forecast,
    parse_generated_forecast,
)
from src.utils.lora import apply_lora_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline evaluation of Qwen2.5-Instruct for time series forecasting")
    parser.add_argument("--output_dir", type=str, default="results/baseline_evaluation", help="Directory to save evaluation results")
    parser.add_argument("--data_path", type=str, default="data/formatted_test.pkl", help="Path to the preprocessed test data file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the local model weights (.safetensors)")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-Instruct", help="Name or path to the tokenizer")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to evaluate (use -1 for all)")
    parser.add_argument("--forecast_length", type=int, default=5, help="Number of steps to forecast")
    parser.add_argument("--context_length", type=int, default=10, help="Number of steps to use as context")
    parser.add_argument("--uncertainty_samples", type=int, default=20, help="Number of samples to generate for uncertainty estimation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of the LoRA model")
    return parser.parse_args()


def prepare_batch_examples(formatted_series_list, context_length=10, forecast_length=5):
    """Prepare multiple context-forecast pairs"""
    contexts = []
    context_data_list = []
    ground_truths = []

    for formatted_series in formatted_series_list:
        # Parse the full series
        series_array = LLMTIMEPreprocessor.parse_formatted_timeseries(formatted_series)

        # Skip if too short
        if len(series_array) <= context_length + forecast_length:
            continue

        # Split into context and ground truth
        context_data = series_array[:context_length]
        ground_truth = series_array[context_length : context_length + forecast_length]

        # Format context as string
        context_str = LLMTIMEPreprocessor.format_timeseries(context_data)

        contexts.append(context_str)
        context_data_list.append(context_data)
        ground_truths.append(ground_truth)

    return contexts, context_data_list, ground_truths


def load_lora_model(model_path, base_model_name="Qwen/Qwen2.5-0.5B-Instruct", rank=4):
    """
    Load a saved LoRA model.

    Args:
        model_path (str): Path to the saved LoRA model
        base_model_name (str): Name of the base model used for training
        rank (int): The rank used during LoRA training

    Returns:
        The loaded model with LoRA weights
    """
    # Instead of loading the base model directly, use your custom loader
    # that handles the LM head bias properly
    logger.info(f"Loading base model from {base_model_name} with proper configuration")
    base_model, _ = load_qwen()  # This adds the lm_head.bias

    # Apply LoRA structure with same rank as used during training
    logger.info(f"Applying LoRA structure with rank {rank}")
    model = apply_lora_to_model(base_model, r=rank, target_modules=["q_proj", "v_proj"])

    # Load the saved state dict (weights) directly
    logger.info(f"Loading saved weights from {model_path}")

    # Try to load the state dict
    if os.path.isdir(model_path):
        state_dict_path = None
        # Search for state dict files
        for filename in ["pytorch_model.bin", "model.safetensors"]:
            if os.path.exists(os.path.join(model_path, filename)):
                state_dict_path = os.path.join(model_path, filename)
                break

        if not state_dict_path:
            # Try to find any .bin or .safetensors file
            for filename in os.listdir(model_path):
                if filename.endswith(".bin") or filename.endswith(".safetensors"):
                    state_dict_path = os.path.join(model_path, filename)
                    logger.info(f"Found model file: {filename}")
                    break

        if state_dict_path:
            # Load the state dict
            if state_dict_path.endswith(".bin"):
                state_dict = torch.load(state_dict_path)
            else:  # .safetensors
                from safetensors.torch import load_file

                state_dict = load_file(state_dict_path)

            # Load state dict with strict=False to allow for architecture differences
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {state_dict_path} with strict=False")
        else:
            logger.warning(f"No model weights found in {model_path}")
    else:
        # It's a direct file path
        if model_path.endswith(".bin"):
            state_dict = torch.load(model_path)
        elif model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            raise ValueError(f"Unknown model file format: {model_path}")

        # Load state dict with strict=False
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from {model_path} with strict=False")

    return model


def main():
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Set style of plots
    plt.style.use("notebooks/mystyle.mplstyle")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {args.output_dir}")

    # Load the preprocessed data
    with open(args.data_path, "rb") as f:
        sample_data = pickle.load(f)

    # Limit number of samples if specified
    if args.num_samples > 0 and args.num_samples < len(sample_data):
        sample_data = sample_data[: args.num_samples]

    logger.info(f"Loaded {len(sample_data)} test sequences")
    logger.info(f"Sample: {sample_data[0][:50]}...")

    # Initialize accelerator
    accelerator = Accelerator()
    logger.info(f"Using device: {accelerator.device}")

    # Load the model
    if args.model_path:
        logger.info(f"Loading custom model from {args.model_path}")
        # Use the load_lora_model function to load the LoRA-adapted model
        model = load_lora_model(model_path=args.model_path, base_model_name="Qwen/Qwen2.5-0.5B-Instruct", rank=args.lora_rank)
        # Load the tokenizer separately
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        # Move model to the appropriate device
        model.to(accelerator.device)
        logger.info("Custom model loaded successfully")
    else:
        logger.info("Loading default Qwen model")
        model, tokenizer = load_qwen()
        model.to(accelerator.device)
        logger.info("Model loaded successfully")

    # Prepare batch examples
    contexts, context_data_list, ground_truths = prepare_batch_examples(
        sample_data, context_length=args.context_length, forecast_length=args.forecast_length
    )
    logger.info(f"Prepared {len(contexts)} context-forecast pairs")

    # Generate forecasts with uncertainty
    logger.info("\nGenerating forecasts with uncertainty...")
    median_forecasts, lower_bounds, upper_bounds, all_samples = batch_generate_forecasts_with_uncertainty(
        model,
        tokenizer,
        contexts,
        forecast_length=args.forecast_length,
        num_samples=args.uncertainty_samples,
        batch_size=args.batch_size,
        return_samples=True,
    )

    np.save(os.path.join(args.output_dir, "median_forecasts.npy"), np.array(median_forecasts, dtype=object))
    np.save(os.path.join(args.output_dir, "lower_bounds.npy"), np.array(lower_bounds, dtype=object))
    np.save(os.path.join(args.output_dir, "upper_bounds.npy"), np.array(upper_bounds, dtype=object))
    np.save(os.path.join(args.output_dir, "all_samples.npy"), np.array(all_samples, dtype=object))

    # Visualize multiple forecasts with uncertainty
    context_data_list_parsed = [parse_generated_forecast(ctx) for ctx in contexts]
    visualize_multiple_forecasts_with_uncertainty(
        median_forecasts,
        lower_bounds,
        upper_bounds,
        ground_truths,
        context_data_list=context_data_list_parsed,
        variable_names=["Prey", "Predator"],
        figsize=(14, 7),
    )
    plt.savefig(os.path.join(args.output_dir, "forecasts_with_uncertainty.pdf"))

    # Evaluate forecasts with uncertainty
    uncertainty_metrics = evaluate_forecasts_with_uncertainty(median_forecasts, lower_bounds, upper_bounds, ground_truths)

    # Print overall metrics
    logger.info("\nBatch Evaluation Metrics:")
    logger.info(f"Mean Squared Error (MSE): {uncertainty_metrics['mse_avg']:.6f}")
    logger.info(f"Root Mean Squared Error (RMSE): {uncertainty_metrics['rmse_avg']:.6f}")
    logger.info(f"Mean Absolute Error (MAE): {uncertainty_metrics['mae_avg']:.6f}")
    logger.info(f"Mean Absolute Percentage Error (MAPE): {uncertainty_metrics['mape_avg']:.2f}%")
    logger.info(f"Coverage: {uncertainty_metrics['coverage']:.2f}")
    logger.info(f"Interval Width: {uncertainty_metrics['interval_width']:.4f}")

    # Save metrics as JSON
    with open(os.path.join(args.output_dir, "uncertainty_metrics.json"), "w") as f:
        json.dump(uncertainty_metrics, f, indent=4)

    # Generate naive forecasts for the same contexts
    logger.info("\nGenerating naive baseline forecasts...")
    naive_forecasts = []
    for context_data in context_data_list:
        naive_pred = naive_forecast(context_data, args.forecast_length)
        naive_forecasts.append(naive_pred)

    # Evaluate naive forecasts
    naive_metrics = evaluate_batch_forecasts(forecasts=naive_forecasts, ground_truths=ground_truths, accelerator=accelerator)

    # Save naive metrics as JSON
    with open(os.path.join(args.output_dir, "naive_metrics.json"), "w") as f:
        json.dump(naive_metrics, f, indent=4)

    # Print comparison of metrics
    logger.info("\nModel vs Naive Baseline Comparison:")
    logger.info(f"{'Metric':<20} {'Model':<10} {'Naive':<10} {'Improvement':<10}")
    logger.info(f"{'-'*50}")
    logger.info(
        f"{'RMSE':<20} {uncertainty_metrics['rmse_avg']:.4f}    {naive_metrics['rmse_avg']:.4f}    {(naive_metrics['rmse_avg'] - uncertainty_metrics['rmse_avg'])/naive_metrics['rmse_avg']*100:.2f}%"
    )
    logger.info(
        f"{'MAE':<20} {uncertainty_metrics['mae_avg']:.4f}    {naive_metrics['mae_avg']:.4f}    {(naive_metrics['mae_avg'] - uncertainty_metrics['mae_avg'])/naive_metrics['mae_avg']*100:.2f}%"
    )
    logger.info(
        f"{'MAPE':<20} {uncertainty_metrics['mape_avg']:.2f}%    {naive_metrics['mape_avg']:.2f}%    {(naive_metrics['mape_avg'] - uncertainty_metrics['mape_avg'])/naive_metrics['mape_avg']*100:.2f}%"
    )

    # Generate naive forecasts with "uncertainty"
    naive_lower = [forecast * 0.9 for forecast in naive_forecasts]
    naive_upper = [forecast * 1.1 for forecast in naive_forecasts]

    # Compare uncertainty metrics
    naive_metrics_unc = evaluate_forecasts_with_uncertainty(naive_forecasts, naive_lower, naive_upper, ground_truths)

    logger.info("\nUncertainty Evaluation Comparison:")
    logger.info(f"{'Metric':<20} {'Model':<10} {'Naive':<10}")
    logger.info(f"{'-'*50}")
    logger.info(f"{'RMSE':<20} {uncertainty_metrics['rmse_avg']:.4f}    {naive_metrics_unc['rmse_avg']:.4f}")
    logger.info(f"{'Coverage':<20} {uncertainty_metrics['coverage']:.2f}    {naive_metrics_unc['coverage']:.2f}")
    logger.info(f"{'Interval Width':<20} {uncertainty_metrics['interval_width']:.4f}    {naive_metrics_unc['interval_width']:.4f}")

    logger.info(f"\nEvaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
