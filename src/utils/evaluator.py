"""
Module: evaluator.py - Time Series Forecast Evaluation

Utilities for evaluating time series forecasts with error metrics and visualization tools.
Supports standard metrics, batch evaluation, and uncertainty quantification.

Key functions:
- Error metrics: calculate_mse, calculate_rmse, calculate_mae, calculate_mape, calculate_smape
- Evaluation: evaluate_forecast, evaluate_batch_forecasts, evaluate_forecasts_with_uncertainty
- Visualization: visualize_forecast, visualize_forecast_with_uncertainty
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_mse(forecast: np.ndarray, ground_truth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate Mean Squared Error between forecast and ground truth.

    Args:
        forecast (np.ndarray): Predicted values with shape (n_steps, n_variables)
        ground_truth (np.ndarray): Actual values with shape (n_steps, n_variables)
        mask (Optional[np.ndarray], optional): Binary mask indicating which values to include
            in calculation. Defaults to None.

    Returns:
        np.ndarray: MSE for each variable with shape (n_variables,)

    Example:
        >>> forecast = np.array([[0.35, 1.35], [0.30, 1.40]])
        >>> ground_truth = np.array([[0.36, 1.33], [0.32, 1.38]])
        >>> calculate_mse(forecast, ground_truth)
        # Output: array([0.00045, 0.00045])
    """
    # Ensure inputs have the same shape
    if forecast.shape != ground_truth.shape:
        raise ValueError(f"Forecast shape {forecast.shape} does not match ground truth shape {ground_truth.shape}")

    # Calculate squared errors
    squared_errors = (forecast - ground_truth) ** 2

    # Apply mask if provided
    if mask is not None:
        if mask.shape != forecast.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match forecast shape {forecast.shape}")
        squared_errors = squared_errors * mask
        # Count non-zero elements in mask for each variable
        counts = np.sum(mask, axis=0)
        mse = np.sum(squared_errors, axis=0) / np.maximum(counts, 1)
    else:
        mse = np.mean(squared_errors, axis=0)

    return mse


def calculate_rmse(forecast: np.ndarray, ground_truth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate Root Mean Squared Error between forecast and ground truth.

    Args:
        forecast (np.ndarray): Predicted values with shape (n_steps, n_variables)
        ground_truth (np.ndarray): Actual values with shape (n_steps, n_variables)
        mask (Optional[np.ndarray], optional): Binary mask indicating which values to include
            in calculation. Defaults to None.

    Returns:
        np.ndarray: RMSE for each variable with shape (n_variables,)

    Example:
        >>> forecast = np.array([[0.35, 1.35], [0.30, 1.40]])
        >>> ground_truth = np.array([[0.36, 1.33], [0.32, 1.38]])
        >>> calculate_rmse(forecast, ground_truth)
        # Output: array([0.02121, 0.02121])
    """
    mse = calculate_mse(forecast, ground_truth, mask)
    return np.sqrt(mse)


def calculate_mae(forecast: np.ndarray, ground_truth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate Mean Absolute Error between forecast and ground truth.

    Args:
        forecast (np.ndarray): Predicted values with shape (n_steps, n_variables)
        ground_truth (np.ndarray): Actual values with shape (n_steps, n_variables)
        mask (Optional[np.ndarray], optional): Binary mask indicating which values to include
            in calculation. Defaults to None.

    Returns:
        np.ndarray: MAE for each variable with shape (n_variables,)

    Example:
        >>> forecast = np.array([[0.35, 1.35], [0.30, 1.40]])
        >>> ground_truth = np.array([[0.36, 1.33], [0.32, 1.38]])
        >>> calculate_mae(forecast, ground_truth)
        # Output: array([0.015, 0.02])
    """
    # Ensure inputs have the same shape
    if forecast.shape != ground_truth.shape:
        raise ValueError(f"Forecast shape {forecast.shape} does not match ground truth shape {ground_truth.shape}")

    # Calculate absolute errors
    absolute_errors = np.abs(forecast - ground_truth)

    # Apply mask if provided
    if mask is not None:
        if mask.shape != forecast.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match forecast shape {forecast.shape}")
        absolute_errors = absolute_errors * mask
        # Count non-zero elements in mask for each variable
        counts = np.sum(mask, axis=0)
        mae = np.sum(absolute_errors, axis=0) / np.maximum(counts, 1)
    else:
        mae = np.mean(absolute_errors, axis=0)

    return mae


def calculate_mape(forecast: np.ndarray, ground_truth: np.ndarray, mask: Optional[np.ndarray] = None, epsilon: float = 1e-10) -> np.ndarray:
    """
    Calculate Mean Absolute Percentage Error between forecast and ground truth.

    Args:
        forecast (np.ndarray): Predicted values with shape (n_steps, n_variables)
        ground_truth (np.ndarray): Actual values with shape (n_steps, n_variables)
        mask (Optional[np.ndarray], optional): Binary mask indicating which values to include
            in calculation. Defaults to None.
        epsilon (float, optional): Small constant to avoid division by zero. Defaults to 1e-10.

    Returns:
        np.ndarray: MAPE for each variable with shape (n_variables,)

    Example:
        >>> forecast = np.array([[0.35, 1.35], [0.30, 1.40]])
        >>> ground_truth = np.array([[0.36, 1.33], [0.32, 1.38]])
        >>> calculate_mape(forecast, ground_truth)
        # Output: array([4.16667, 1.47929])  # percent
    """
    # Ensure inputs have the same shape
    if forecast.shape != ground_truth.shape:
        raise ValueError(f"Forecast shape {forecast.shape} does not match ground truth shape {ground_truth.shape}")

    # Add epsilon to ground truth to avoid division by zero
    # Only where ground truth is zero or very close to zero
    safe_ground_truth = np.where(np.abs(ground_truth) < epsilon, np.sign(ground_truth) * epsilon, ground_truth)

    # Calculate percentage errors
    percentage_errors = 100 * np.abs((forecast - ground_truth) / safe_ground_truth)

    # Apply mask if provided
    if mask is not None:
        if mask.shape != forecast.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match forecast shape {forecast.shape}")
        percentage_errors = percentage_errors * mask
        # Count non-zero elements in mask for each variable
        counts = np.sum(mask, axis=0)
        mape = np.sum(percentage_errors, axis=0) / np.maximum(counts, 1)
    else:
        mape = np.mean(percentage_errors, axis=0)

    return mape


def calculate_smape(
    forecast: np.ndarray, ground_truth: np.ndarray, mask: Optional[np.ndarray] = None, epsilon: float = 1e-10
) -> np.ndarray:
    """
    Calculate Symmetric Mean Absolute Percentage Error between forecast and ground truth.

    Args:
        forecast (np.ndarray): Predicted values with shape (n_steps, n_variables)
        ground_truth (np.ndarray): Actual values with shape (n_steps, n_variables)
        mask (Optional[np.ndarray], optional): Binary mask indicating which values to include
            in calculation. Defaults to None.
        epsilon (float, optional): Small constant to avoid division by zero. Defaults to 1e-10.

    Returns:
        np.ndarray: SMAPE for each variable with shape (n_variables,)

    Example:
        >>> forecast = np.array([[0.35, 1.35], [0.30, 1.40]])
        >>> ground_truth = np.array([[0.36, 1.33], [0.32, 1.38]])
        >>> calculate_smape(forecast, ground_truth)
        # Output: array([4.11523, 1.47757])  # percent
    """
    # Ensure inputs have the same shape
    if forecast.shape != ground_truth.shape:
        raise ValueError(f"Forecast shape {forecast.shape} does not match ground truth shape {ground_truth.shape}")

    # Calculate numerator (absolute difference) and denominator (sum of absolute values)
    numerator = np.abs(forecast - ground_truth)
    denominator = np.abs(forecast) + np.abs(ground_truth) + epsilon

    # Calculate symmetric percentage errors (multiplied by 200 as per the standard formula)
    symmetric_percentage_errors = 200 * numerator / denominator

    # Apply mask if provided
    if mask is not None:
        if mask.shape != forecast.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match forecast shape {forecast.shape}")
        symmetric_percentage_errors = symmetric_percentage_errors * mask
        # Count non-zero elements in mask for each variable
        counts = np.sum(mask, axis=0)
        smape = np.sum(symmetric_percentage_errors, axis=0) / np.maximum(counts, 1)
    else:
        smape = np.mean(symmetric_percentage_errors, axis=0)

    return smape


def evaluate_forecast(
    forecast: np.ndarray, ground_truth: np.ndarray, mask: Optional[np.ndarray] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate multiple error metrics between forecast and ground truth.

    Args:
        forecast (np.ndarray): Predicted values with shape (n_steps, n_variables)
        ground_truth (np.ndarray): Actual values with shape (n_steps, n_variables)
        mask (Optional[np.ndarray], optional): Binary mask indicating which values to include
            in calculation. Defaults to None.

    Returns:
        Dict[str, Union[float, np.ndarray]]: Dictionary of error metrics including
            individual metrics (mse, rmse, mae, mape, smape) and their averages
    """
    # Calculate all metrics
    mse = calculate_mse(forecast, ground_truth, mask)
    rmse = np.sqrt(mse)
    mae = calculate_mae(forecast, ground_truth, mask)
    mape = calculate_mape(forecast, ground_truth, mask)
    smape = calculate_smape(forecast, ground_truth, mask)

    # Compile metrics into a dictionary
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "mse_avg": np.mean(mse),
        "rmse_avg": np.mean(rmse),
        "mae_avg": np.mean(mae),
        "mape_avg": np.mean(mape),
        "smape_avg": np.mean(smape),
    }

    logger.info(f"Forecast evaluation - RMSE: {metrics['rmse_avg']:.4f}, MAE: {metrics['mae_avg']:.4f}, MAPE: {metrics['mape_avg']:.2f}%")
    return metrics


def evaluate_batch_forecasts(
    forecasts: List[np.ndarray], ground_truths: List[np.ndarray], accelerator: Optional[Accelerator] = None
) -> Dict[str, float]:
    """
    Evaluate forecasts against ground truths across multiple time series.

    Args:
        forecasts (List[np.ndarray]): List of forecasted values
        ground_truths (List[np.ndarray]): List of ground truth values
        accelerator (Optional[Accelerator], optional): Accelerator for distributed evaluation.
            Defaults to None.

    Returns:
        Dict[str, float]: Dictionary of averaged error metrics across all series
    """
    # Ensure we have the same number of forecasts and ground truths
    if len(forecasts) != len(ground_truths):
        raise ValueError(f"Number of forecasts ({len(forecasts)}) does not match number of ground truths ({len(ground_truths)})")

    # Calculate metrics for each pair
    all_metrics = []
    for forecast, ground_truth in zip(forecasts, ground_truths):
        metrics = evaluate_forecast(forecast, ground_truth)
        all_metrics.append(metrics)

    # Calculate average metrics across all series
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], np.ndarray):
            # For per-variable metrics, first average across variables in each series
            avg_metrics[key] = np.mean([np.mean(m[key]) for m in all_metrics])
        else:
            # For already averaged metrics, just average across series
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # If using accelerator, gather and average metrics across all processes
    if accelerator:
        for key in avg_metrics.keys():
            value = torch.tensor([avg_metrics[key]], device=accelerator.device, dtype=torch.float32)
            gathered_values = accelerator.gather(value)
            avg_metrics[key] = gathered_values.mean().item()

    logger.info(
        f"Batch evaluation - RMSE: {avg_metrics['rmse_avg']:.4f}, MAE: {avg_metrics['mae_avg']:.4f}, MAPE: {avg_metrics['mape_avg']:.2f}%"
    )
    return avg_metrics


def visualize_forecast(
    forecast: np.ndarray,
    ground_truth: np.ndarray,
    context_data: Optional[np.ndarray] = None,
    variable_names: Optional[List[str]] = None,
    title: str = "Time Series Forecast",
    figsize: Tuple[int, int] = (6, 4),
) -> plt.Figure:
    """
    Visualize forecast against ground truth with optional historical context.

    Args:
        forecast (np.ndarray): Predicted values with shape (n_steps, n_variables)
        ground_truth (np.ndarray): Actual values with shape (n_steps, n_variables)
        context_data (Optional[np.ndarray], optional): Historical context data with
            shape (context_length, n_variables). Defaults to None.
        variable_names (Optional[List[str]], optional): Names of the variables for the legend.
            Defaults to None.
        title (str, optional): Plot title. Defaults to "Time Series Forecast".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (6, 4).

    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Number of variables
    n_variables = forecast.shape[1]
    if variable_names is None:
        variable_names = [f"Variable {i+1}" for i in range(n_variables)]

    # Create x-axis values
    forecast_steps = np.arange(forecast.shape[0])

    # If context is provided, plot it first
    if context_data is not None:
        context_steps = np.arange(-context_data.shape[0], 0)
        for i in range(n_variables):
            ax.plot(
                context_steps,
                context_data[:, i],
                "o-",
                color=f"C{i}",
                alpha=0.5,
                label=f"{variable_names[i]} (Context)",
            )

    # Plot ground truth
    for i in range(n_variables):
        ax.plot(forecast_steps, ground_truth[:, i], "o-", color=f"C{i}", label=f"{variable_names[i]} (Actual)")

    # Plot forecast
    for i in range(n_variables):
        ax.plot(forecast_steps, forecast[:, i], "x--", color=f"C{i}", alpha=0.8, label=f"{variable_names[i]} (Forecast)")

    # Add a vertical line separating context from forecast if context is provided
    if context_data is not None:
        ax.axvline(x=-0.5, color="k", linestyle="--", alpha=0.5)

    # Add labels and title
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    return fig


def evaluate_forecasts_with_uncertainty(
    median_forecasts: List[np.ndarray],
    lower_bounds: List[np.ndarray],
    upper_bounds: List[np.ndarray],
    ground_truths: List[np.ndarray],
    accelerator: Optional[Accelerator] = None,
) -> Dict[str, float]:
    """
    Evaluate forecasts with uncertainty bounds against ground truths.

    Args:
        median_forecasts (List[np.ndarray]): List of median forecasted values
        lower_bounds (List[np.ndarray]): List of lower bound values
        upper_bounds (List[np.ndarray]): List of upper bound values
        ground_truths (List[np.ndarray]): List of ground truth values
        accelerator (Optional[Accelerator], optional): Accelerator for distributed evaluation.

    Returns:
        Dict[str, float]: Dictionary of metrics including standard error metrics,
            coverage (% of ground truth within bounds), and interval_width
    """
    # Ensure we have the same number of forecasts and ground truths
    if len(median_forecasts) != len(ground_truths):
        raise ValueError(f"Number of forecasts ({len(median_forecasts)}) does not match number of ground truths ({len(ground_truths)})")

    if len(lower_bounds) != len(median_forecasts) or len(upper_bounds) != len(median_forecasts):
        raise ValueError("Number of lower/upper bounds does not match number of median forecasts")

    # Calculate standard metrics for median forecasts
    standard_metrics = evaluate_batch_forecasts(median_forecasts, ground_truths, accelerator)

    # Calculate uncertainty-specific metrics
    coverage_scores = []
    interval_widths = []

    for lower, upper, truth in zip(lower_bounds, upper_bounds, ground_truths):
        # Calculate coverage (percentage of ground truth points within the interval)
        within_interval = np.logical_and(truth >= lower, truth <= upper)
        coverage = np.mean(within_interval)
        coverage_scores.append(coverage)

        # Calculate average interval width
        width = np.mean(upper - lower)
        interval_widths.append(width)

    # Add uncertainty metrics to the results
    uncertainty_metrics = {
        "coverage": np.mean(coverage_scores),
        "interval_width": np.mean(interval_widths),
    }

    # Combine all metrics
    all_metrics = {**standard_metrics, **uncertainty_metrics}

    logger.info(
        f"Evaluation with uncertainty - RMSE: {all_metrics['rmse_avg']:.4f}, Coverage: {all_metrics['coverage']:.2f}, Interval width: {all_metrics['interval_width']:.4f}"
    )

    return all_metrics


def visualize_multiple_forecasts_with_uncertainty(
    median_forecasts: List[np.ndarray],
    lower_bounds: List[np.ndarray],
    upper_bounds: List[np.ndarray],
    ground_truths: List[np.ndarray],
    context_data_list: Optional[List[np.ndarray]] = None,
    variable_names: Optional[List[str]] = None,
    series_names: Optional[List[str]] = None,
    max_series: int = 4,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Create a grid of plots for multiple forecasts with uncertainty bounds.

    Args:
        median_forecasts (List[np.ndarray]): List of median forecasts
        lower_bounds (List[np.ndarray]): List of lower bounds
        upper_bounds (List[np.ndarray]): List of upper bounds
        ground_truths (List[np.ndarray]): List of ground truth values
        context_data_list (Optional[List[np.ndarray]], optional): List of historical context data. Defaults to None.
        variable_names (Optional[List[str]], optional): Names of variables. Defaults to None.
        series_names (Optional[List[str]], optional): Names for each series. Defaults to None.
        max_series (int, optional): Maximum number of series to plot. Defaults to 4.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 10).

    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Limit the number of series to plot
    n_series = min(len(median_forecasts), max_series)
    n_variables = median_forecasts[0].shape[1]

    # Default variable and series names
    if variable_names is None:
        variable_names = [f"Variable {i+1}" for i in range(n_variables)]

    if series_names is None:
        series_names = [f"Series {i+1}" for i in range(n_series)]

    # Create figure with grid of subplots (variables as rows, series as columns)
    fig, axes = plt.subplots(n_variables, n_series, figsize=figsize, sharex=True)

    # Handle case when there's only one variable or one series
    if n_variables == 1 and n_series == 1:
        axes = np.array([[axes]])
    elif n_variables == 1:
        axes = np.array([axes])
    elif n_series == 1:
        axes = np.array([[ax] for ax in axes])

    # Plot each series and variable
    for i in range(n_series):
        median = median_forecasts[i]
        lower = lower_bounds[i]
        upper = upper_bounds[i]
        truth = ground_truths[i]

        # Get context if available
        context = None
        if context_data_list and i < len(context_data_list):
            context = context_data_list[i]

        # Create x-axis values
        forecast_steps = np.arange(median.shape[0])
        context_steps = None
        if context is not None:
            context_steps = np.arange(-context.shape[0], 0)

        # Plot for each variable
        for j in range(n_variables):
            ax = axes[j, i]

            # Plot context if provided
            if context is not None:
                ax.plot(context_steps, context[:, j], "o-", color=f"C{j}", alpha=0.5)

            # Plot ground truth
            ax.plot(forecast_steps, truth[:, j], "o-", color=f"C{j}")

            # Plot median forecast
            ax.plot(forecast_steps, median[:, j], "x--", color=f"C{j}", alpha=0.8)

            # Plot uncertainty bounds as shaded area
            ax.fill_between(forecast_steps, lower[:, j], upper[:, j], color=f"C{j}", alpha=0.2)

            # Add a vertical line separating context from forecast if context is provided
            if context is not None:
                ax.axvline(x=-0.5, color="k", linestyle="--", alpha=0.5)

            # Add labels for first column and last row
            if i == 0:
                ax.set_ylabel(variable_names[j])
            if j == n_variables - 1:
                ax.set_xlabel("Time Steps")

            # Add series name as title for top row
            if j == 0:
                ax.set_title(series_names[i])

            ax.grid(True, alpha=0.3)

    # Create a common legend
    handles, labels = [], []
    handles.append(plt.Line2D([0], [0], color="k", marker="o", linestyle="-"))
    labels.append("Actual Values")
    handles.append(plt.Line2D([0], [0], color="k", marker="x", linestyle="--"))
    labels.append("Median Forecast")
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="k", alpha=0.2))
    labels.append("80% Confidence Interval")
    if context is not None:
        handles.append(plt.Line2D([0], [0], color="k", marker="o", linestyle="-", alpha=0.5))
        labels.append("Historical Data")

    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for the legend

    return fig


def visualize_forecast_with_uncertainty(
    median_forecast: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    ground_truth: np.ndarray,
    context_data: Optional[np.ndarray] = None,
    variable_names: Optional[List[str]] = None,
    title: str = "Time Series Forecast with Uncertainty",
    figsize: Tuple[int, int] = (10, 6),
    show_percentiles: bool = True,
) -> plt.Figure:
    """
    Visualize the forecast with uncertainty bounds against the ground truth.

    Args:
        median_forecast: Median predicted values with shape (n_steps, n_variables)
        lower_bound: Lower bound (e.g., 10th percentile) with shape (n_steps, n_variables)
        upper_bound: Upper bound (e.g., 90th percentile) with shape (n_steps, n_variables)
        ground_truth: Actual values with shape (n_steps, n_variables)
        context_data: Historical context data with shape (context_length, n_variables)
        variable_names: Names of the variables for the legend
        title: Plot title
        figsize: Figure size
        show_percentiles: Whether to show percentile bounds in the legend

    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Create figure with subplots (one per variable)
    n_variables = median_forecast.shape[1]
    fig, axes = plt.subplots(n_variables, 1, figsize=figsize, sharex=True)

    # If only one variable, make axes a list for consistent indexing
    if n_variables == 1:
        axes = [axes]

    # Variable names
    if variable_names is None:
        variable_names = [f"Variable {i+1}" for i in range(n_variables)]

    # Create x-axis values
    forecast_steps = np.arange(median_forecast.shape[0])

    # Prepare for context data
    context_steps = None
    if context_data is not None:
        context_steps = np.arange(-context_data.shape[0], 0)

    # Plot for each variable
    for i in range(n_variables):
        ax = axes[i]

        # Plot context if provided
        if context_data is not None:
            ax.plot(context_steps, context_data[:, i], "o-", color=f"C{i}", alpha=0.5, label="Historical Data")

        # Plot ground truth
        ax.plot(forecast_steps, ground_truth[:, i], "o-", color=f"C{i}", label="Actual Values")

        # Plot median forecast
        ax.plot(forecast_steps, median_forecast[:, i], "x--", color=f"C{i}", alpha=0.8, label="Median Forecast")

        # Plot uncertainty bounds as shaded area
        label_text = "80% Confidence Interval" if show_percentiles else None
        ax.fill_between(forecast_steps, lower_bound[:, i], upper_bound[:, i], color=f"C{i}", alpha=0.2, label=label_text)

        # Add a vertical line separating context from forecast if context is provided
        if context_data is not None:
            ax.axvline(x=-0.5, color="k", linestyle="--", alpha=0.5)

        # Add labels and legend
        ax.set_ylabel(variable_names[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # Add shared x-axis label
    axes[-1].set_xlabel("Time Step")

    # Add title
    fig.suptitle(title)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the title

    return fig
