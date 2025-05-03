"""
Module: flops_calculator.py - FLOPs Calculation for Language Models

Functions to estimate computational requirements (FLOPs) for language model inference
and training. Supports Qwen2.5 architecture analysis, comparison between model sizes,
and calculation of LoRA adaptation overhead.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def forward_flops(
    B: int,
    S: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    vocab_size: int,
) -> int:
    """
    Calculate FLOPs for a model's forward pass, broken down by component.

    Args:
        B (int): Batch size
        S (int): Sequence length
        hidden_size (int): Hidden state dimension
        intermediate_size (int): MLP intermediate layer dimension
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        vocab_size (int): Size of the vocabulary

    Returns:
        dict: FLOPs breakdown for each component and total forward pass
    """
    # Fallback values if B or S are None (should not happen in normal operation)
    if B is None:
        B = 2  # Default small batch size
        logger.warning("Batch size is None in FLOPS calculation. Using default value of 2.")
    if S is None:
        S = 128  # Default sequence length
        logger.warning("Sequence length is None in FLOPS calculation. Using default value of 128.")

    forward_flops_dict = {}

    # Embedding lookup: B*S*V @ V*H conceptually, but implemented as lookup = 0 FLOPS
    # Adding positional encodings: B*S*H additions
    forward_flops_dict["initial_embedding"] = B * S * hidden_size

    # All transformer layers: L * layer FLOPS
    layer_flops_dict = layer_flops(B, S, hidden_size, intermediate_size, num_heads, head_dim)
    forward_flops_dict["layer_flops_per_layer"] = layer_flops_dict
    forward_flops_dict["layer_flops_total"] = num_layers * layer_flops_dict["total"]

    # Final RMSNorm: same as layer norm
    forward_flops_dict["final_norm"] = B * S * (4 * hidden_size + 11)

    # LM Head: B*S*H @ H*V with bias = B*S*V*2H
    forward_flops_dict["lm_head"] = B * S * vocab_size * 2 * hidden_size

    # Calculate total FLOPS for the forward pass
    forward_flops_dict["total"] = (
        forward_flops_dict["initial_embedding"]
        + forward_flops_dict["layer_flops_total"]
        + forward_flops_dict["final_norm"]
        + forward_flops_dict["lm_head"]
    )

    logger.debug(f"Forward pass total FLOPs: {forward_flops_dict['total']:,} for batch size {B}, sequence length {S}")
    return forward_flops_dict


def layer_flops(B: int, S: int, hidden_size: int, intermediate_size: int, num_heads: int, head_dim: int) -> int:
    """
    Calculate FLOPs for a single transformer layer.

    Args:
        B (int): Batch size
        S (int): Sequence length
        hidden_size (int): Hidden state dimension
        intermediate_size (int): MLP intermediate layer dimension
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head

    Returns:
        dict: FLOPs breakdown by component (attention, MLP, etc.)
    """
    flops = {}

    # Input RMSNorm: (H mult for squares + 1 for scaling + H div + H mult by g) + (H-1 add + 1 for eps) + 10 for sqrt
    # = 4H + 11 FLOPS per token
    flops["input_layernorm"] = B * S * (4 * hidden_size + 11)

    # Self-Attention (simplified as standard MHA)
    # Q, K, V projections: each B*S*H @ H*H with bias = B*S*H*2H
    flops["q_proj"] = B * S * hidden_size * 2 * hidden_size
    flops["k_proj"] = B * S * hidden_size * 2 * hidden_size
    flops["v_proj"] = B * S * hidden_size * 2 * hidden_size
    # Scores: N_h * (B*S*D_h @ D_h*S) = N_h * B*S*S*(2D_h - 1)
    flops["scores"] = num_heads * B * S * S * (2 * head_dim - 1)
    # Scaling: N_h * B*S*S multiplications
    flops["scaling"] = num_heads * B * S * S
    # Softmax: N_h * B*S rows, each S exp (10) + (S-1) adds + S divs = 12S - 1
    flops["softmax"] = num_heads * B * S * (12 * S - 1)
    # Weighted Sum: N_h * (B*S*S @ S*D_h) = N_h * B*S*D_h*(2S - 1)
    flops["weighted_sum"] = num_heads * B * S * head_dim * (2 * S - 1)
    # Output proj: B*S*H @ H*H, no bias = B*S*H*(2H - 1)
    flops["o_proj"] = B * S * hidden_size * (2 * hidden_size - 1)

    # Residual: B*S*H additions
    flops["add_residual1"] = B * S * hidden_size

    # Post-Attention RMSNorm: same as input
    flops["post_attention_layernorm"] = B * S * (4 * hidden_size + 11)

    # MLP
    # Gate: B*S*H @ H*I, no bias = B*S*I*(2H - 1)
    flops["gate_proj"] = B * S * intermediate_size * (2 * hidden_size - 1)
    # Up: same as gate
    flops["up_proj"] = B * S * intermediate_size * (2 * hidden_size - 1)
    # SwiGLU SiLU: 13 FLOPS per element (exp 10 + add 1 + div 1 + mult 1)
    flops["activation"] = B * S * intermediate_size * 13
    # Element-wise mult: B*S*I
    flops["elementwise_mult"] = B * S * intermediate_size
    # Down: B*S*I @ I*H, no bias = B*S*H*(2I - 1)
    flops["down_proj"] = B * S * hidden_size * (2 * intermediate_size - 1)

    # Residual: B*S*H additions
    flops["add_residual2"] = B * S * hidden_size

    # Calculate Total FLOPS
    # Attention Total
    flops["attention_total"] = (
        flops["q_proj"]
        + flops["k_proj"]
        + flops["v_proj"]
        + flops["scores"]
        + flops["scaling"]
        + flops["softmax"]
        + flops["weighted_sum"]
        + flops["o_proj"]
    )
    # MLP Total
    flops["mlp_total"] = flops["gate_proj"] + flops["up_proj"] + flops["activation"] + flops["elementwise_mult"] + flops["down_proj"]
    # Layer Total
    flops["total"] = (
        flops["input_layernorm"]
        + flops["attention_total"]
        + flops["add_residual1"]
        + flops["post_attention_layernorm"]
        + flops["mlp_total"]
        + flops["add_residual2"]
    )

    logger.debug(f"Single layer FLOPs: Attention={flops['attention_total']:,}, MLP={flops['mlp_total']:,}")
    return flops


def calculate_flops(B: int, S: int, is_training: bool = True, model_size: str = "0.5B") -> dict:
    """
    Calculate total FLOPs for a model's forward pass and optional backward pass.

    Args:
        B (int): Batch size
        S (int): Sequence length
        is_training (bool): Include backward pass (3x forward) if True
        model_size (str): Size variant of model (e.g., "0.5B")

    Returns:
        dict: FLOPs breakdown and total for the specified scenario

    Raises:
        ValueError: If unsupported model size is provided
    """
    # Model hyperparameters based on size
    model_configs = {
        "0.5B": {
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_layers": 24,
            "num_heads": 14,
            "head_dim": 64,
            "vocab_size": 151936,
        }
    }

    if model_size not in model_configs:
        logger.error(f"Invalid model size: {model_size}")
        raise ValueError(f"Model size {model_size} not supported. Choose from: {list(model_configs.keys())}")

    config = model_configs[model_size]
    logger.debug(f"Calculating FLOPs for {model_size} model with batch_size={B}, seq_length={S}, is_training={is_training}")

    # Compute forward pass FLOPS
    forward_dict = forward_flops(
        B,
        S,
        config["hidden_size"],
        config["intermediate_size"],
        config["num_layers"],
        config["num_heads"],
        config["head_dim"],
        config["vocab_size"],
    )

    # Create result dictionary
    result = {
        "model_size": model_size,
        "batch_size": B,
        "sequence_length": S,
        "is_training": is_training,
        "model_config": config,
        "forward_pass": forward_dict,
    }

    # Add backward pass if training
    if is_training:
        # Backward pass is assumed to be 2x the forward pass
        result["backward_pass"] = 2 * forward_dict["total"]
        result["total_flops"] = 3 * forward_dict["total"]
        logger.info(
            f"Training FLOPs: forward={forward_dict['total']:,}, backward={result['backward_pass']:,}, total={result['total_flops']:,}"
        )
    else:
        result["total_flops"] = forward_dict["total"]
        logger.info(f"Inference FLOPs: total={result['total_flops']:,}")

    # Add per-token FLOPS
    result["flops_per_token"] = result["total_flops"] / (B * S)

    return result


def calculate_lora_flops(
    batch_size: int,
    seq_length: int,
    r: int,
    steps: int,
    model_size: str = "0.5B",
    target_modules: list = None,
    is_training: bool = True,
) -> dict:
    """
    Calculate FLOPs for LoRA adaptation, including additional low-rank computations.

    Args:
        batch_size (int): Batch size for training/inference
        seq_length (int): Sequence length of inputs
        r (int): Rank of LoRA adaptation
        steps (int): Number of optimization steps or inference passes
        model_size (str): Size variant of model, defaults to "0.5B"
        target_modules (list): Module types to adapt with LoRA, defaults to ["q_proj", "v_proj"]
        is_training (bool): Include backward pass if True, defaults to True

    Returns:
        dict: FLOPs breakdown for standard ops, LoRA overhead, and combined total

    Raises:
        ValueError: If unsupported model size is provided
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # Model configuration
    model_configs = {
        "0.5B": {
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_layers": 24,
            "num_heads": 14,
            "head_dim": 64,
            "vocab_size": 151936,
        }
    }

    if model_size not in model_configs:
        logger.error(f"Invalid model size: {model_size}")
        raise ValueError(f"Model size {model_size} not supported. Choose from: {list(model_configs.keys())}")

    config = model_configs[model_size]
    hidden_dim = config["hidden_size"]
    num_layers = config["num_layers"]

    # Initialize FLOPS dictionary
    lora_flops = {}

    # Number of LoRA modules
    num_modules = len(target_modules) * num_layers
    mode = "training" if is_training else "inference"
    logger.info(f"Calculating LoRA FLOPs for {num_modules} modules with rank {r}, over {steps} steps in {mode} mode")

    # Standard training/inference FLOPS
    std_flops = calculate_flops(batch_size, seq_length, is_training=is_training, model_size=model_size)

    # LoRA additional computations per module
    # A x: (B*S*H) @ (H*r) = B*S*r*(2H - 1)
    # B (A x): (B*S*r) @ (r*H) = B*S*H*(2r - 1)
    # Addition: B*S*H
    # Total: B*S*(4rH - r)
    lora_additional_per_module = batch_size * seq_length * (4 * r * hidden_dim - r)

    # Total LoRA additional FLOPS per forward pass
    lora_additional_forward = lora_additional_per_module * num_modules

    # Backward pass: 2x forward (only for training)
    lora_additional_backward = 2 * lora_additional_forward if is_training else 0

    # Total LoRA additional FLOPS per step
    lora_additional_per_step = lora_additional_forward + lora_additional_backward

    # Total LoRA FLOPS across all steps
    lora_total = lora_additional_per_step * steps

    # Organize results
    lora_flops["lora_additional_per_module"] = lora_additional_per_module
    lora_flops["lora_modules_count"] = num_modules
    lora_flops["lora_additional_forward"] = lora_additional_forward

    if is_training:
        lora_flops["lora_additional_backward"] = lora_additional_backward

    lora_flops["lora_additional_per_step"] = lora_additional_per_step
    lora_flops["lora_total_additional"] = lora_total

    # Standard FLOPS
    lora_flops["standard_forward"] = std_flops["forward_pass"]["total"]

    if is_training:
        lora_flops["standard_backward"] = std_flops["backward_pass"]

    lora_flops["standard_per_step"] = std_flops["total_flops"]
    lora_flops["standard_total"] = std_flops["total_flops"] * steps

    # Combined FLOPS
    lora_flops["combined_per_step"] = std_flops["total_flops"] + lora_additional_per_step
    lora_flops["combined_total"] = lora_flops["standard_total"] + lora_flops["lora_total_additional"]

    # Log summary
    lora_overhead_percent = (lora_flops["lora_additional_per_step"] / std_flops["total_flops"]) * 100
    process_type = "training" if is_training else "inference"
    logger.info(f"LoRA overhead: {lora_overhead_percent:.2f}% FLOPs per step compared to standard {process_type}")
    logger.info(f"Total LoRA {process_type} FLOPs: {lora_flops['combined_total']:,} for {steps} steps")

    return lora_flops
