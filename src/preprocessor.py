"""
Module: preprocessor.py - Time Series Data Preprocessing for Language Models

Tools for converting numerical time series data to text format for language models.
Implements the LLMTIME format where variables are comma-separated and timesteps are
semicolon-separated (e.g., "0.5,1.2;0.6,1.1").

Key functionalities:
- Loading and preprocessing time series data
- Scaling numerical values for language model compatibility
- Converting between text and numerical representations
- Creating train/validation splits
- Tokenization for model training

Classes:
    LLMTIMEPreprocessor: Main preprocessing class with static methods for data manipulation

Functions:
    load_and_preprocess: Helper function to load, preprocess, and split data

Example:
    ```python
    # Load and preprocess data
    train_texts, val_texts = load_and_preprocess(
        "data.h5", scaling_factor=10.0, decimal_places=2
    )

    # Format a numpy array as text
    series = np.array([[1.5, 2.3], [1.6, 2.2]])
    formatted = LLMTIMEPreprocessor.format_timeseries(series)
    # Output: "1.50,2.30;1.60,2.20"

    # Parse text back to array
    parsed = LLMTIMEPreprocessor.parse_formatted_timeseries(formatted)
    ```
"""

import logging
import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LLMTIMEPreprocessor:
    """
    Preprocessor for time series data using the LLMTIME scheme.

    Converts numerical time series data to text format for language models,
    with comma-separated variables and semicolon-separated timesteps.

    Key capabilities:
    - Loading time series data from HDF5 files
    - Scaling and rounding numerical values
    - Formatting arrays as text with specified decimal places
    - Converting text back to numerical arrays
    - Tokenizing text for language model input
    - Creating training/validation splits

    Format example: "0.25,1.50;0.27,1.47" (2 timesteps with 2 variables)
    """

    def __init__(self):
        """
        Initialize the LLMTIME preprocessor.
        """
        logger.info("Initialized preprocessor")

    @staticmethod
    def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load time series data from an HDF5 file.

        Args:
            file_path (str): Path to the HDF5 file containing the time series data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Trajectories array of shape
            (n_trajectories, n_timesteps, n_variables) and corresponding time points.

        Raises:
            FileNotFoundError: If the specified file_path does not exist.
        """
        if not os.path.exists(file_path):
            logger.error(f"Data file not found at {file_path}")
            raise FileNotFoundError(f"Data file not found at {file_path}")

        try:
            with h5py.File(file_path, "r") as f:
                trajectories = f["trajectories"][:]
                time_points = f["time"][:]

            logger.info(f"Loaded data with shape: {trajectories.shape}")
            return trajectories, time_points
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise

    @staticmethod
    def scale_and_round(trajectories: np.ndarray, scaling_factor: float = 1.0, decimal_places: int = 2) -> np.ndarray:
        """
        Scale the trajectories by a factor and round to specified decimal places.

        Args:
            trajectories (np.ndarray): Input trajectories array of shape
                (n_trajectories, n_timesteps, n_variables).
            scaling_factor (float, optional): Factor to divide the values by. Defaults to 1.0.
            decimal_places (int, optional): Number of decimal places to round to. Defaults to 2.

        Returns:
            np.ndarray: Scaled and rounded trajectories with the same shape as the input.
        """
        scaled_trajectories = trajectories / scaling_factor
        rounded_trajectories = np.round(scaled_trajectories, decimal_places)
        logger.info(f"Scaled trajectories by factor {scaling_factor} and rounded to {decimal_places} decimal places")
        logger.debug(
            f"Original range: [{trajectories.min():.4f}, {trajectories.max():.4f}], "
            f"Scaled range: [{rounded_trajectories.min():.4f}, {rounded_trajectories.max():.4f}]"
        )
        return rounded_trajectories

    @staticmethod
    def format_timeseries_batch(trajectories: np.ndarray, decimal_places: int = 2) -> List[str]:
        """
        Format multiple time series into LLMTIME text format.

        Args:
            trajectories (np.ndarray): Input trajectories array of shape
                (n_trajectories, n_timesteps, n_variables).
            decimal_places (int, optional): Number of decimal places to include.
                Defaults to 2.

        Returns:
            List[str]: List of formatted strings, one for each trajectory.
        """
        formatted_strings = []

        for sample in trajectories:
            sample_str = LLMTIMEPreprocessor.format_timeseries(sample, decimal_places)
            formatted_strings.append(sample_str)

        logger.info(f"Formatted {len(formatted_strings)} time series into LLMTIME format")
        return formatted_strings

    @staticmethod
    def format_timeseries(timeseries: np.ndarray, decimal_places: int = 2) -> str:
        """
        Format a single time series into LLMTIME text format.

        Args:
            timeseries (np.ndarray): Input time series array of shape
                (n_timesteps, n_variables).
            decimal_places (int, optional): Number of decimal places to include.
                Defaults to 2.

        Returns:
            str: Formatted string with variables separated by commas and
                timesteps by semicolons.
        """
        timesteps = []
        for timestep in timeseries:
            timestep_str = LLMTIMEPreprocessor.format_timestep(timestep, decimal_places)
            timesteps.append(timestep_str)

        # Join timesteps with semicolons
        sample_str = ";".join(timesteps)
        return sample_str

    @staticmethod
    def format_timestep(timestep: np.ndarray, decimal_places: int = 2) -> str:
        """
        Format a single timestep into a comma-separated string.

        Args:
            timestep (np.ndarray): Input array of variable values with shape
                (n_variables,).
            decimal_places (int, optional): Number of decimal places to include.
                Defaults to 2.

        Returns:
            str: Comma-separated string of variable values.
        """
        # Format each value with exactly the specified decimal places and join with commas
        format_str = f"{{:.{decimal_places}f}}"
        timestep_str = ",".join(format_str.format(val) for val in timestep)
        return timestep_str

    @staticmethod
    def format_timeseries_slice(timeseries: np.ndarray, start_idx: int = 0, end_idx: Optional[int] = None, decimal_places: int = 2) -> str:
        """
        Format a slice of a time series into LLMTIME text format.

        Args:
            timeseries (np.ndarray): Input time series array of shape
                (n_timesteps, n_variables).
            start_idx (int, optional): Starting index of the slice. Defaults to 0.
            end_idx (Optional[int], optional): Ending index (exclusive). Defaults to None.
            decimal_places (int, optional): Number of decimal places to include. Defaults to 2.

        Returns:
            str: Formatted string for the specified slice.
        """
        # Extract the slice
        if end_idx is None:
            timeseries_slice = timeseries[start_idx:]
        else:
            timeseries_slice = timeseries[start_idx:end_idx]

        # Format the slice
        return LLMTIMEPreprocessor.format_timeseries(timeseries_slice, decimal_places)

    @staticmethod
    def batch_tokenize(samples: List[str], tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> List[List[int]]:
        """
        Tokenize multiple formatted time series samples.

        Args:
            samples (List[str]): List of formatted time series strings.
            tokenizer_name (str, optional): Name or path of the tokenizer.
                Defaults to "Qwen/Qwen2.5-0.5B-Instruct".

        Returns:
            List[List[int]]: List of tokenized samples (token ID lists).
        """
        tokenized_samples = []
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        for sample in tqdm(samples, desc="Tokenizing samples"):
            tokenized_sample = LLMTIMEPreprocessor.tokenize_sample(sample, tokenizer)
            tokenized_samples.append(tokenized_sample)

        return tokenized_samples

    @staticmethod
    def tokenize_sample(sample: str, tokenizer=None, tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> List[int]:
        """
        Tokenize a single formatted time series sample.

        Args:
            sample (str): Formatted time series string.
            tokenizer: Pre-loaded tokenizer. If None, will load using tokenizer_name.
            tokenizer_name (str, optional): Tokenizer name/path.
                Defaults to "Qwen/Qwen2.5-0.5B-Instruct".

        Returns:
            List[int]: List of token IDs.
        """
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        tokenized_sample = tokenizer.encode(sample, add_special_tokens=False)
        return tokenized_sample

    @staticmethod
    def decode_sample(tokenized_sample: List[int], tokenizer=None, tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> str:
        """
        Decode a tokenized sample back into text.

        Args:
            tokenized_sample (List[int]): List of token IDs.
            tokenizer: Pre-loaded tokenizer. If None, will load using tokenizer_name.
            tokenizer_name (str, optional): Tokenizer name/path.
                Defaults to "Qwen/Qwen2.5-0.5B-Instruct".

        Returns:
            str: Decoded text representation.
        """
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        decoded_sample = tokenizer.decode(tokenized_sample)
        return decoded_sample

    @staticmethod
    def parse_formatted_timeseries(formatted_str: str) -> np.ndarray:
        """
        Parse a formatted time series string back into numeric values.

        Args:
            formatted_str (str): Formatted string in LLMTIME format.

        Returns:
            np.ndarray: Array of shape (n_timesteps, n_variables) with numeric values.
        """
        # Split the string by semicolons to get timesteps
        timestep_strings = formatted_str.split(";")

        # Parse each timestep
        parsed_data = []
        for timestep_str in timestep_strings:
            if not timestep_str.strip():
                continue  # Skip empty timesteps

            parsed_timestep = LLMTIMEPreprocessor.parse_formatted_timestep(timestep_str)
            parsed_data.append(parsed_timestep)

        return np.array(parsed_data)

    @staticmethod
    def parse_formatted_timestep(timestep_str: str) -> List[float]:
        """
        Parse a formatted timestep string back into numeric values.

        Args:
            timestep_str (str): Comma-separated string of values.

        Returns:
            List[float]: List of parsed float values.

        Raises:
            ValueError: If no valid numeric values are found.
        """
        try:
            # Remove any non-numeric characters except commas, periods, minus signs
            # This is a more aggressive cleaning approach for problematic outputs
            clean_str = "".join(c for c in timestep_str if c.isdigit() or c in ",.+-")

            # Split the timestep string by commas and convert to float
            values = []
            for val in clean_str.split(","):
                if val.strip():  # Skip empty values
                    try:
                        values.append(float(val))
                    except ValueError:
                        continue  # Skip values that can't be converted

            # If we got no valid values, raise an error
            if not values:
                raise ValueError(f"No valid numeric values found in '{timestep_str}'")

            return values
        except Exception as e:
            raise ValueError(f"Error parsing timestep '{timestep_str}': {e}")

    @staticmethod
    def combine_context_and_forecast(context_str: str, forecast_str: str) -> str:
        """
        Combine formatted context and forecast strings.

        Args:
            context_str (str): Formatted context string.
            forecast_str (str): Formatted forecast string.

        Returns:
            str: Combined string with proper semicolon separator.
        """
        # Remove trailing semicolon from context if present
        if context_str.endswith(";"):
            context_str = context_str[:-1]

        # Remove leading semicolon from forecast if present
        if forecast_str.startswith(";"):
            forecast_str = forecast_str[1:]

        # Combine with a semicolon separator
        if context_str and forecast_str:
            return f"{context_str};{forecast_str}"
        elif context_str:
            return context_str
        else:
            return forecast_str

    @staticmethod
    def split_train_val(trajectories: np.ndarray, val_split: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split trajectories into training and validation sets.

        Args:
            trajectories (np.ndarray): Input trajectories array of shape
                (n_trajectories, n_timesteps, n_variables).
            val_split (float, optional): Fraction for validation. Defaults to 0.2.
            random_state (int, optional): Random seed. Defaults to 42.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Training and validation trajectory arrays.
        """
        np.random.seed(random_state)
        n_trajectories = trajectories.shape[0]
        indices = np.random.permutation(n_trajectories)
        val_size = int(n_trajectories * val_split)

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_trajectories = trajectories[train_indices]
        val_trajectories = trajectories[val_indices]

        logger.info(f"Split data into {train_trajectories.shape[0]} training and {val_trajectories.shape[0]} validation trajectories")
        return train_trajectories, val_trajectories


def load_and_preprocess(
    file_path: str, scaling_factor: float = 10.0, decimal_places: int = 2, train_ratio: float = 0.8, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Load, scale, format, and split time series data in one operation.

    Args:
        file_path (str): Path to the HDF5 file containing the time series data.
        scaling_factor (float, optional): Factor to divide values by. Defaults to 10.0.
        decimal_places (int, optional): Decimal places to round to. Defaults to 2.
        train_ratio (float, optional): Fraction to use for training. Defaults to 0.8.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[List[str], List[str]]: Training and validation formatted strings.

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logger.warning(f"File not found at {file_path}, attempting to locate in workspace")
        # Try to find the file in the current directory or parent directories
        base_name = os.path.basename(file_path)
        for root, _, files in os.walk("/workspace"):
            if base_name in files:
                file_path = os.path.join(root, base_name)
                logger.info(f"Found file at {file_path}")
                break
        else:
            logger.error(f"Data file not found: {base_name}")
            raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load the data
    trajectories, time_points = LLMTIMEPreprocessor.load_data(file_path)

    # Scale and round the data
    scaled_trajectories = LLMTIMEPreprocessor.scale_and_round(trajectories, scaling_factor=scaling_factor, decimal_places=decimal_places)

    # Format the data as text
    logger.info(f"Formatting {len(scaled_trajectories)} trajectories with {decimal_places} decimal places")
    formatted_texts = LLMTIMEPreprocessor.format_timeseries_batch(scaled_trajectories, decimal_places=decimal_places)

    # Split into train and validation sets
    np.random.seed(seed)
    indices = np.random.permutation(len(formatted_texts))
    split_idx = int(len(formatted_texts) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_texts = [formatted_texts[i] for i in train_indices]
    val_texts = [formatted_texts[i] for i in val_indices]

    logger.info(f"Split data into {len(train_texts)} training and {len(val_texts)} validation samples")
    return train_texts, val_texts
