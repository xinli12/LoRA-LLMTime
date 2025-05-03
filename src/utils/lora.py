"""
Module: lora.py - Low-Rank Adaptation for Parameter-Efficient Fine-tuning

Implementation of LoRA (Low-Rank Adaptation) for efficient fine-tuning of
language models with minimal trainable parameters. Includes tools for applying
LoRA to specific model components, training loops, and data processing utilities.
"""

import logging
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils.flops_calculator import calculate_lora_flops

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# LoRA implementation
class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation for linear layers.

    Wraps a linear layer with trainable low-rank adaptation matrices while
    freezing the original weights.

    Args:
        original_linear (nn.Linear): Original linear layer to be wrapped
        r (int): Rank of the low-rank adaptation
        alpha (int, optional): Scaling factor. Default is r

    Attributes:
        original_linear (nn.Linear): Original frozen linear layer
        r (int): Rank of adaptation
        alpha (int): Scaling factor
        A (nn.Parameter): Low-rank matrix A with shape (r, in_dim)
        B (nn.Parameter): Low-rank matrix B with shape (out_dim, r)
    """

    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)

        self.original_linear = original_linear

        # Freeze the original linear weights and bias
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # Input and output dimensions from the original linear layer
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        self.r = r  # Rank of the low-rank approximation
        self.alpha = alpha if alpha else r  # Scaling factor

        device = original_linear.weight.device  # Ensure same device

        # Define trainable low-rank matrices A and B
        # A maps input to a lower-dimensional space (r x in_dim)
        # B maps from low-dimensional space to output (out_dim x r)
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))

        # He initialization for A (to help with training stability)
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

        logger.debug(f"Created LoRA layer with rank {r}, input dim {in_dim}, output dim {out_dim}")

    def forward(self, x):
        """
        Forward pass: original output + scaled low-rank adaptation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Compute output from the frozen original linear layer
        base_out = self.original_linear(x)

        # Compute the low-rank adaptation output
        lora_out = (x @ self.A.T) @ self.B.T

        # Return the sum of the original output and scaled LoRA output
        return base_out + lora_out * (self.alpha / self.r)


def apply_lora_to_model(model, r: int, target_modules: List[str] = None, alpha: int = None):
    """
    Apply LoRA to specified modules in a model.

    Args:
        model: Transformer model to be adapted
        r (int): Rank for LoRA adaptation (smaller r = fewer parameters)
        target_modules (List[str], optional): Module types to wrap with LoRA.
            Defaults to ["q_proj", "v_proj"].
        alpha (int, optional): Scaling factor. Defaults to r.

    Returns:
        Model with LoRA applied to target modules
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]  # Default modules to adapt for Qwen2.5
        logger.info(f"Using default target modules: {target_modules}")

    alpha = alpha if alpha is not None else r
    logger.info(f"Applying LoRA with rank {r} and alpha {alpha}")

    # Count the number of trainable parameters before applying LoRA
    orig_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Original trainable parameters: {orig_params:,}")

    lora_module_count = 0

    # Traverse the model and replace target modules with LoRA versions
    for layer_idx, layer in enumerate(model.model.layers):
        for name, module in layer.self_attn.named_children():
            if name in target_modules and isinstance(module, nn.Linear):
                # Replace with LoRALinear
                lora_layer = LoRALinear(module, r=r, alpha=alpha)
                setattr(layer.self_attn, name, lora_layer)
                lora_module_count += 1
                logger.debug(f"Applied LoRA to layer {layer_idx}, module {name}")

    # Count trainable parameters after applying LoRA
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Log parameter efficiency and number of modules wrapped
    logger.info(f"Applied LoRA with rank {r} to {lora_module_count} {target_modules} modules")
    logger.info(f"Trainable parameters: {lora_params:,} (original: {orig_params:,}, ratio: {lora_params/max(1, orig_params):.6f})")

    if lora_module_count == 0:
        logger.warning("No modules were wrapped with LoRA! Check target_modules and model structure")

    return model


def process_sequences(texts, tokenizer, max_length=512, stride=256):
    """
    Process text sequences with sliding windows for efficient training data creation.

    Args:
        texts (List[str]): Input text sequences
        tokenizer: Tokenizer for converting text to token IDs
        max_length (int): Maximum sequence length for each chunk. Default: 512
        stride (int): Step size for sliding window. Default: 256

    Returns:
        torch.Tensor: Tokenized sequences with shape (num_chunks, max_length)
    """
    all_input_ids = []

    logger.info(f"Processing {len(texts)} sequences with max_length={max_length}, stride={stride}")

    for i, text in enumerate(tqdm(texts, desc="Processing sequences")):
        # Apply tokenization scheme to the text
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]

        # Create sliding windows to further divide the data into chunks
        chunks_count = 0
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat(
                    [
                        chunk,
                        torch.full((max_length - len(chunk),), tokenizer.pad_token_id),
                    ]
                )
            all_input_ids.append(chunk)
            chunks_count += 1

        if i % 100 == 0:
            logger.debug(f"Created {chunks_count} chunks from sequence of length {len(seq_ids)}")

    logger.info(f"Created {len(all_input_ids)} total chunks from {len(texts)} sequences")
    return torch.stack(all_input_ids)


def train_lora(
    model,
    train_loader,
    optimizer,
    accelerator,
    max_steps: int = 10000,
    eval_loader=None,
    eval_steps: int = 500,
    logging_steps: int = 100,
    save_steps: int = 0,
    output_dir: str = "results/lora",
    model_name: str = "lora_model",
    save_best: bool = True,
    save_final: bool = False,
):
    """
    Train a model with LoRA adaptation.

    Args:
        model: Model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        accelerator: Accelerator for distributed training
        max_steps (int): Maximum training steps. Default: 10000
        eval_loader: DataLoader for evaluation. Default: None
        eval_steps (int): Steps between evaluations. Default: 500
        logging_steps (int): Steps between logging. Default: 100
        save_steps (int): Steps between saving checkpoints (0 to disable). Default: 0
        output_dir (str): Directory to save models. Default: "results/lora"
        model_name (str): Name for saved model. Default: "lora_model"
        save_best (bool): Save model with best eval loss. Default: True
        save_final (bool): Save model after training completes. Default: False

    Returns:
        Dict: Training statistics including losses, steps, and FLOPs usage
    """
    # Safely capture batch size and sequence length before accelerator preparation
    # Use a default if batch_size attribute is None
    batch_size = getattr(train_loader, "batch_size", None)
    if batch_size is None:
        # Try to determine from the first batch
        for batch in train_loader:
            batch_size = batch[0].shape[0]
            break

    # Safely get sequence length from dataset
    try:
        seq_length = train_loader.dataset.tensors[0].shape[1]
    except (AttributeError, IndexError):
        # Try to determine from the first batch
        for batch in train_loader:
            seq_length = batch[0].shape[1]
            break

    # Ensure we have valid values
    if batch_size is None or seq_length is None:
        logger.warning("Could not determine batch_size or seq_length automatically. Using defaults.")
        batch_size = batch_size or 2  # Default batch size
        seq_length = seq_length or 128  # Default sequence length

    logger.info(f"Training with batch_size={batch_size}, seq_length={seq_length}")
    logger.debug(f"FLOPS calculation using batch_size={batch_size}, seq_length={seq_length}")

    # Prepare the model for training
    model.train()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize variables for tracking
    steps = 0
    best_eval_loss = float("inf")
    train_losses = []
    eval_losses = []
    evaluation_steps = []  # Renamed from eval_steps to avoid name collision
    flops_used = 0

    # Get the actual LoRA rank from the model
    lora_rank = None
    # Search for LoRA modules to extract the rank
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_rank = module.r
            break

    # Use actual rank or fall back to default if not found
    if lora_rank is None:
        logger.warning("Could not find LoRA modules in the model. Using default rank=16 for FLOPS calculation.")
        lora_rank = 16
    else:
        logger.info(f"Using actual LoRA rank={lora_rank} for FLOPS calculation")

    # FLOPS per optimization step (forward + backward)
    flops_per_step = calculate_lora_flops(
        batch_size=batch_size,
        seq_length=seq_length,
        r=lora_rank,  # Use the actual rank instead of hardcoded value
        steps=max_steps,
        model_size="0.5B",
        target_modules=["q_proj", "v_proj"],
    )["combined_total"]
    logger.info(f"Estimated FLOPs per step: {flops_per_step:,}")

    # Training loop
    logger.info(f"Starting training for {max_steps} steps")
    progress_bar = tqdm(total=max_steps, desc="Training")
    while steps < max_steps:
        for (batch,) in train_loader:
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch, labels=batch)
            loss = outputs.loss

            # Backward pass and optimization
            accelerator.backward(loss)
            optimizer.step()

            # Track training loss
            train_losses.append(loss.item())

            # Update progress bar
            steps += 1
            flops_used += flops_per_step
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

            # Log training progress
            if steps % logging_steps == 0:
                logger.info(f"Step {steps}: Train loss = {loss.item():.4f}, FLOPS used = {flops_used:.2e}")

            # Evaluate model
            if eval_loader is not None and steps % eval_steps == 0:
                logger.info(f"Evaluating at step {steps}...")
                eval_loss = evaluate_model(model, eval_loader, accelerator)
                eval_losses.append(eval_loss["loss"])
                evaluation_steps.append(steps)  # Use renamed variable
                logger.info(f"Step {steps}: Eval loss = {eval_loss['loss']:.4f}, perplexity = {eval_loss['perplexity']:.4f}")

                if eval_loss["loss"] < best_eval_loss:
                    best_eval_loss = eval_loss["loss"]
                    # Save best model only if save_best is True
                    if save_best:
                        logger.info(f"New best eval loss: {best_eval_loss:.4f} at step {steps}, saving model")
                        accelerator.save_model(model, f"{output_dir}/{model_name}_best")

            # Save checkpoint only if save_steps > 0
            if save_steps > 0 and steps % save_steps == 0:
                logger.info(f"Saving checkpoint at step {steps}")
                accelerator.save_model(model, f"{output_dir}/{model_name}_step{steps}")

            # Check if we've reached max steps
            if steps >= max_steps:
                break

    progress_bar.close()

    # Save final model only if save_final is True
    if save_final:
        logger.info("Training complete, saving final model")
        accelerator.save_model(model, f"{output_dir}/{model_name}_final")

    # Return training results
    results = {
        "steps": steps,
        "train_losses": train_losses,
        "eval_losses": eval_losses if eval_loader is not None else None,
        "eval_steps": evaluation_steps if eval_loader is not None else None,  # Use renamed variable
        "best_eval_loss": best_eval_loss if eval_loader is not None else None,
        "flops_used": flops_used,
    }

    logger.info(f"Training completed with {steps} steps. Best eval loss: {best_eval_loss:.4f}")
    return results


def evaluate_model(model, eval_loader, accelerator):
    """
    Evaluate a model on validation data.

    Args:
        model: Model to evaluate
        eval_loader: DataLoader for evaluation data
        accelerator: Accelerator for distributed evaluation

    Returns:
        dict: Evaluation metrics containing 'loss' and 'perplexity'
    """
    model.eval()
    total_loss = 0
    total_samples = 0

    logger.debug("Starting model evaluation")

    with torch.no_grad():
        for (batch,) in tqdm(eval_loader, desc="Evaluating", position=0, leave=True):
            outputs = model(batch, labels=batch)
            loss = outputs.loss

            # Get batch size for proper weighting
            batch_size = batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    model.train()

    # Calculate average loss
    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    logger.debug(f"Evaluation complete on {total_samples} samples: loss={avg_loss:.4f}, perplexity={perplexity:.4f}")

    # Return metrics dictionary
    return {"loss": avg_loss, "perplexity": perplexity}


def create_train_val_loaders(train_texts, val_texts, tokenizer, max_ctx_length, batch_size):
    """
    Create DataLoaders for training and validation data.

    Args:
        train_texts (List[str]): Training text sequences
        val_texts (List[str]): Validation text sequences
        tokenizer: Tokenizer to use
        max_ctx_length (int): Maximum context length for each chunk
        batch_size (int): Batch size for training

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders
    """
    logger.info(f"Creating data loaders with max_ctx_length={max_ctx_length}, batch_size={batch_size}")

    # Process sequences
    logger.info(f"Processing {len(train_texts)} training sequences")
    train_input_ids = process_sequences(train_texts, tokenizer, max_ctx_length, stride=max_ctx_length // 2)

    logger.info(f"Processing {len(val_texts)} validation sequences")
    val_input_ids = process_sequences(val_texts, tokenizer, max_ctx_length, stride=max_ctx_length)

    # Create datasets
    train_dataset = TensorDataset(train_input_ids)
    val_dataset = TensorDataset(val_input_ids)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Created train loader with {len(train_dataset)} samples and val loader with {len(val_dataset)} samples")
    return train_loader, val_loader
