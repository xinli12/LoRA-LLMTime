"""
Module: generator.py - Time Series Forecasting with Language Models

Utilities for generating time series forecasts using language models.
Handles single forecasts, uncertainty quantification, batch processing, and error handling.

Uses LLMTIME format: variables separated by commas, timesteps by semicolons.
Example: "0.5,1.2;0.6,1.1" represents 2 timesteps with 2 variables each.

Key functions:
- generate_forecast: Create single forecast from context
- generate_forecast_with_uncertainty: Generate forecasts with confidence intervals
- batch_generate_forecasts: Process multiple contexts efficiently
- parse_generated_forecast: Convert text to numerical arrays
- naive_forecast: Generate baseline persistence forecasts

Main Functions:
    generate_forecast: Generate a single forecast from historical context
    generate_forecast_with_uncertainty: Generate forecasts with confidence intervals
    batch_generate_forecasts: Generate forecasts for multiple contexts in batches
    batch_generate_forecasts_with_uncertainty: Batch forecasting with uncertainty
    parse_generated_forecast: Convert generated text into numerical arrays
    naive_forecast: Generate persistence-based baseline forecasts
    complete_forecast: Extend incomplete forecasts to expected length
    extract_forecast: Separate forecast data from historical context

Helper Functions:
    prepare_forecast_inputs: Tokenize context for model input

Classes:
    SemicolonCounter: Custom stopping criteria for generation control

Example:
    ```python
    # Generate a forecast
    forecast_text = generate_forecast(
        model, tokenizer,
        context_str="0.5,1.2;0.6,1.1",
        forecast_length=5,
        temperature=0.7
    )

    # Parse forecast into numerical values
    forecast_values = parse_generated_forecast(forecast_text)

    # Generate forecasts with uncertainty bounds
    median, lower, upper, samples = generate_forecast_with_uncertainty(
        model, tokenizer,
        context_str="0.5,1.2;0.6,1.1",
        forecast_length=5,
        num_samples=20,
        return_samples=True
    )
    ```
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers.generation.stopping_criteria import StoppingCriteriaList

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Regular expression for finding numbers in text (used in parse_generated_forecast)
FLOAT_PATTERN = re.compile(r"-?\d+\.?\d*")


def prepare_forecast_inputs(context: str, tokenizer, max_length: int = 512, device: str = None) -> Dict[str, torch.Tensor]:
    """
    Prepare token inputs for time series forecasting.

    Args:
        context (str): Formatted time series context in LLMTIME format
        tokenizer: Tokenizer for encoding the context
        max_length (int, optional): Maximum sequence length. Defaults to 512
        device (str, optional): Device for tensors. Defaults to None

    Returns:
        Dict[str, torch.Tensor]: Dictionary of model inputs including input_ids
    """
    # Tokenize the context
    inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False, padding=False)

    # Move to device if specified
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    logger.debug(f"Prepared forecast inputs with {inputs['input_ids'].shape[1]} tokens")
    return inputs


def generate_forecast(
    model,
    tokenizer,
    context_str: str = None,
    forecast_length: int = 5,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 20,
    accelerator: Optional[Accelerator] = None,
    verbose: bool = False,
) -> str:
    """
    Generate a time series forecast using a language model.

    Args:
        model: Language model for forecasting
        tokenizer: Tokenizer for the language model
        context_str (str): Formatted context in LLMTIME format
        forecast_length (int): Number of timesteps to forecast. Defaults to 5
        max_new_tokens (int): Maximum new tokens to generate. Defaults to 100
        temperature (float): Sampling temperature. Defaults to 0.7
        top_k (int): Tokens to consider for sampling. Defaults to 20
        accelerator (Optional[Accelerator]): For distributed inference. Defaults to None
        verbose (bool): Print detailed output. Defaults to False

    Returns:
        str: Generated forecast in LLMTIME format
    """
    if verbose:
        logger.info(f"Generating forecast for {forecast_length} steps")
    else:
        logger.debug(f"Generating forecast for {forecast_length} steps")

    device = model.device

    logger.debug(f"Tokenizing context string of length {len(context_str)}")
    inputs = prepare_forecast_inputs(context_str, tokenizer, device=device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)

    if accelerator is not None:
        model, input_ids, attention_mask = accelerator.prepare(model, input_ids, attention_mask)

    # Get semicolon token ID for stopping logic
    semicolon_token_id = tokenizer.encode(";")[0]

    # Create a faster stopping criteria based on semicolon count
    class SemicolonCounter:
        def __init__(self, semicolon_id, forecast_length):
            self.semicolon_count = 0
            self.forecast_length = forecast_length
            self.semicolon_id = semicolon_id
            self.previous_input_length = None

        def __call__(self, input_ids, scores, **kwargs):
            # Initialize on first call
            if self.previous_input_length is None:
                self.previous_input_length = input_ids.shape[1]
                return False

            # Fast path: Check only the new tokens since last call
            new_tokens = input_ids[0, self.previous_input_length :]
            # Count semicolons in new tokens (using tensor operations instead of loop)
            self.semicolon_count += torch.sum(new_tokens == self.semicolon_id).item()
            self.previous_input_length = input_ids.shape[1]

            # Stop if we've reached the desired forecast length
            return self.semicolon_count >= self.forecast_length

    semicolon_counter = SemicolonCounter(semicolon_token_id, forecast_length)
    stopping_criteria = StoppingCriteriaList([semicolon_counter])

    if verbose:
        logger.info(f"Generating forecast with length {forecast_length}, temperature {temperature}, top_k {top_k}")
    else:
        logger.debug(f"Generating forecast with length {forecast_length}, temperature {temperature}, top_k {top_k}")

    # Generate forecast
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)

    # Ensure the generated text doesn't end with a semicolon
    if generated_text.endswith(";"):
        generated_text = generated_text[:-1]

    # Ensure we have exactly the number of forecasts specified by forecast_length
    semicolons = generated_text.count(";")
    if semicolons > forecast_length:
        # If we have too many semicolons, truncate the text
        last_pos = -1
        for _ in range(forecast_length):
            last_pos = generated_text.find(";", last_pos + 1)
        if last_pos != -1:
            generated_text = generated_text[: last_pos + 1]  # +1 to include the semicolon

    if verbose:
        logger.info(f"Generated forecast: {generated_text}")
    else:
        logger.debug(f"Generated forecast with {generated_text.count(';')} timesteps")

    return generated_text


def parse_generated_forecast(generated_text: str, expected_var_count: int = 2) -> np.ndarray:
    """
    Parse a forecast text string into a numerical array.

    Args:
        generated_text (str): Forecast text in LLMTIME format (e.g., "0.35,1.35;0.30,1.40")
        expected_var_count (int, optional): Expected variables per timestep. Defaults to 2

    Returns:
        np.ndarray: Array with shape (n_timesteps, n_variables)
    """
    if not generated_text or not generated_text.strip():
        logger.warning("Empty forecast text received")
        return np.array([])

    # Split the generated text by semicolons to get timesteps
    timesteps = generated_text.strip().split(";")
    parsed_data = []

    for timestep_idx, timestep in enumerate(timesteps):
        timestep = timestep.strip()
        if not timestep:
            continue

        # Split by comma to get variables for this timestep
        variables = timestep.split(",")
        parsed_timestep = []

        for var_idx, var in enumerate(variables):
            var = var.strip()
            try:
                # Extract numbers using the pre-compiled regex pattern
                number_matches = FLOAT_PATTERN.findall(var)

                if number_matches:
                    # Use the first valid number found
                    parsed_timestep.append(float(number_matches[0]))
                else:
                    # If no valid number found, try direct conversion
                    parsed_timestep.append(float(var))
            except ValueError:
                logger.warning(f"Couldn't parse value at timestep {timestep_idx}, variable {var_idx}: '{var}'")
                continue

        # Handle timesteps with wrong number of variables
        if parsed_timestep:
            if len(parsed_timestep) < expected_var_count:
                # Pad with the last value or zeros
                last_value = parsed_timestep[-1] if parsed_timestep else 0.0
                parsed_timestep.extend([last_value] * (expected_var_count - len(parsed_timestep)))
            elif len(parsed_timestep) > expected_var_count:
                # Truncate to expected number
                parsed_timestep = parsed_timestep[:expected_var_count]
                logger.warning(f"Timestep {timestep_idx} truncated from {len(variables)} to {expected_var_count} variables")

            parsed_data.append(parsed_timestep)

    if not parsed_data:
        logger.warning("No valid timesteps could be parsed from the generated forecast")
        return np.array([])

    return np.array(parsed_data)


def batch_generate_forecasts(
    model,
    tokenizer,
    context_str: List[str],
    forecast_length: int = 10,
    batch_size: int = 4,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 20,
    accelerator: Optional[Accelerator] = None,
    verbose: bool = False,
) -> List[str]:
    """
    Generate forecasts for multiple context sequences.

    Args:
        model: Language model for forecasting
        tokenizer: Tokenizer for the model
        context_str (List[str]): List of context strings in LLMTIME format
        forecast_length (int): Timesteps to forecast. Defaults to 10
        batch_size (int): Contexts to process per batch. Defaults to 4
        max_new_tokens (int): Maximum tokens to generate. Defaults to 100
        temperature (float): Sampling temperature. Defaults to 0.7
        top_k (int): Tokens to consider for sampling. Defaults to 20
        accelerator (Optional[Accelerator]): For distributed inference. Defaults to None
        verbose (bool): Print detailed output. Defaults to False

    Returns:
        List[str]: List of forecast strings in LLMTIME format
    """
    forecasts = []

    # Set model to evaluation mode
    model.eval()

    # Process contexts in batches
    for i in tqdm(range(0, len(context_str), batch_size), desc="Generating forecasts"):
        batch_contexts = context_str[i : i + batch_size]
        batch_forecasts = []

        for context in batch_contexts:
            forecast = generate_forecast(
                model,
                tokenizer,
                context_str=context,
                forecast_length=forecast_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                accelerator=accelerator,
                verbose=verbose,
            )
            batch_forecasts.append(forecast)

        forecasts.extend(batch_forecasts)

    logger.info(f"Generated {len(forecasts)} forecasts")

    return forecasts


def naive_forecast(context_array: np.ndarray, forecast_length: int) -> np.ndarray:
    """
    Generate a persistence-based forecast by repeating the last observed values.

    Args:
        context_array (np.ndarray): Historical data with shape (n_timesteps, n_variables)
        forecast_length (int): Number of future timesteps to forecast

    Returns:
        np.ndarray: Naive forecast with shape (forecast_length, n_variables)
    """
    last_value = context_array[-1:]
    return np.repeat(last_value, forecast_length, axis=0)


def complete_forecast(forecast: np.ndarray, expected_length: int) -> np.ndarray:
    """
    Extend an incomplete forecast to the expected length.

    Args:
        forecast (np.ndarray): Generated forecast with shape (n_timesteps, n_variables)
        expected_length (int): Expected number of forecast timesteps

    Returns:
        np.ndarray: Completed forecast with shape (expected_length, n_variables)
    """
    if forecast.shape[0] >= expected_length:
        # If forecast is already long enough, truncate to expected length
        return forecast[:expected_length]

    # Calculate how many timesteps are missing
    missing_steps = expected_length - forecast.shape[0]

    if forecast.shape[0] == 0:
        logger.warning("Empty forecast received, cannot complete")
        return np.zeros((expected_length, forecast.shape[1] if forecast.size > 0 else 2))

    # Generate naive forecast for the missing steps using the last values
    extension = naive_forecast(forecast, missing_steps)

    # Concatenate the original forecast with the extension
    completed_forecast = np.vstack([forecast, extension])

    logger.info(f"Forecast completed: extended from {forecast.shape[0]} to {completed_forecast.shape[0]} timesteps")

    return completed_forecast


def extract_forecast(generated_text: str, context: str) -> str:
    """
    Extract only the forecast part from generated text.

    Args:
        generated_text (str): Full generated text (context + forecast)
        context (str): Original context provided for generation

    Returns:
        str: Extracted forecast part without the context
    """
    # Handle case where context is not at the beginning of the generated text
    if context not in generated_text:
        logger.warning("Context not found in generated text. Returning full text.")
        return generated_text

    # Find where the context ends
    forecast_start = generated_text.find(context) + len(context)

    # Extract the forecast part
    forecast_text = generated_text[forecast_start:].strip()

    # Add semicolon if needed
    if forecast_text and not forecast_text.startswith(";") and not context.endswith(";"):
        forecast_text = ";" + forecast_text

    # Remove leading semicolon if present
    if forecast_text.startswith(";"):
        forecast_text = forecast_text[1:]

    return forecast_text


def generate_forecast_with_uncertainty(
    model,
    tokenizer,
    context_str: str,
    forecast_length: int = 5,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 20,
    num_samples: int = 10,
    accelerator: Optional[Accelerator] = None,
    return_samples: bool = False,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate forecast with uncertainty estimates through Monte Carlo sampling.

    Args:
        model: Language model for forecasting
        tokenizer: Tokenizer for the model
        context_str (str): Context in LLMTIME format
        forecast_length (int): Timesteps to forecast. Defaults to 5
        max_new_tokens (int): Maximum tokens to generate. Defaults to 100
        temperature (float): Sampling temperature. Defaults to 0.7
        top_k (int): Tokens to consider for sampling. Defaults to 20
        num_samples (int): Number of samples for uncertainty. Defaults to 10
        accelerator (Optional[Accelerator]): For distributed inference. Defaults to None
        return_samples (bool): Return all samples. Defaults to False
        verbose (bool): Print detailed output. Defaults to False

    Returns:
        If return_samples=False:
            np.ndarray: Median forecast with shape (forecast_length, n_variables)
        If return_samples=True:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Median, lower bound, upper bound, and all samples
    """
    if verbose:
        logger.info(f"Generating {num_samples} forecast samples for uncertainty estimation")

    forecast_samples = []

    # Generate multiple forecast samples
    for i in range(num_samples):
        if verbose:
            logger.info(f"Generating forecast sample {i+1}/{num_samples}")
        forecast_text = generate_forecast(
            model,
            tokenizer,
            context_str=context_str,
            forecast_length=forecast_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            accelerator=accelerator,
            verbose=False,
        )

        # Parse the generated forecast text to get numerical values
        forecast_array = parse_generated_forecast(forecast_text)

        # Skip empty forecasts
        if len(forecast_array) == 0:
            if verbose:
                logger.warning(f"Sample {i+1} produced an empty forecast")
            continue

        # Make sure all forecasts have the same length by using complete_forecast
        if forecast_array.shape[0] < forecast_length:
            forecast_array = complete_forecast(forecast_array, forecast_length)
        elif forecast_array.shape[0] > forecast_length:
            forecast_array = forecast_array[:forecast_length]

        forecast_samples.append(forecast_array)

    # Make sure we have at least one valid forecast
    if not forecast_samples:
        logger.warning("No valid forecasts were generated")
        # Return a naive forecast based on the context
        context_array = parse_generated_forecast(context_str)
        median_forecast = naive_forecast(context_array, forecast_length)
        if return_samples:
            return median_forecast, median_forecast, median_forecast, np.expand_dims(median_forecast, axis=0)
        else:
            return median_forecast

    # Convert list of forecast arrays to a 3D array [num_samples, forecast_length, n_variables]
    forecast_samples_array = np.stack(forecast_samples, axis=0)

    # Calculate median forecast (along the samples dimension)
    median_forecast = np.median(forecast_samples_array, axis=0)

    # Calculate 10th and 90th percentiles for uncertainty bounds
    lower_bound = np.percentile(forecast_samples_array, 10, axis=0)
    upper_bound = np.percentile(forecast_samples_array, 90, axis=0)

    if verbose:
        logger.info(f"Generated {len(forecast_samples)} valid forecast samples")

    if return_samples:
        return median_forecast, lower_bound, upper_bound, forecast_samples_array
    else:
        return median_forecast


def batch_generate_forecasts_with_uncertainty(
    model,
    tokenizer,
    context_str: List[str],
    forecast_length: int = 10,
    batch_size: int = 4,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 20,
    num_samples: int = 10,
    accelerator: Optional[Accelerator] = None,
    return_samples: bool = False,
    verbose: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
    """
    Generate forecasts with uncertainty for multiple contexts.

    Args:
        model: Language model for forecasting
        tokenizer: Tokenizer for the model
        context_str (List[str]): List of context strings in LLMTIME format
        forecast_length (int): Timesteps to forecast. Defaults to 10
        batch_size (int): Contexts to process per batch. Defaults to 4
        max_new_tokens (int): Maximum tokens to generate. Defaults to 100
        temperature (float): Sampling temperature. Defaults to 0.7
        top_k (int): Tokens to consider for sampling. Defaults to 20
        num_samples (int): Number of samples for uncertainty. Defaults to 10
        accelerator (Optional[Accelerator]): For distributed inference. Defaults to None
        return_samples (bool): Return all samples. Defaults to False
        verbose (bool): Print detailed output. Defaults to False

    Returns:
        If return_samples=False:
            List[np.ndarray]: List of median forecasts
        If return_samples=True:
            Tuple containing lists of median forecasts, lower bounds, upper bounds, and samples
    """
    # Set model to evaluation mode
    model.eval()

    median_forecasts = []
    lower_bounds = []
    upper_bounds = []
    all_samples = []

    # Process contexts in batches for the outer loop
    for i in tqdm(range(0, len(context_str), batch_size), desc="Generating batch forecasts with uncertainty"):
        batch_contexts = context_str[i : i + batch_size]
        batch_medians = []
        batch_lowers = []
        batch_uppers = []
        batch_samples = []

        # Process each context in the batch
        for context in batch_contexts:
            # Generate forecasts with uncertainty
            result = generate_forecast_with_uncertainty(
                model,
                tokenizer,
                context_str=context,
                forecast_length=forecast_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                num_samples=num_samples,
                accelerator=accelerator,
                return_samples=True,
                verbose=verbose,
            )

            median_forecast, lower_bound, upper_bound, samples = result

            batch_medians.append(median_forecast)
            batch_lowers.append(lower_bound)
            batch_uppers.append(upper_bound)
            batch_samples.append(samples)

        median_forecasts.extend(batch_medians)
        lower_bounds.extend(batch_lowers)
        upper_bounds.extend(batch_uppers)
        all_samples.extend(batch_samples)

    logger.info(f"Generated forecasts with uncertainty for {len(median_forecasts)} contexts")

    if return_samples:
        return median_forecasts, lower_bounds, upper_bounds, all_samples
    else:
        return median_forecasts
