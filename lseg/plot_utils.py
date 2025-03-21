import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(predictions: pd.Series, actuals: pd.Series, title: str = "Model vs Actual"):
    """
    Plot predicted prices vs actual prices over the evaluation period.

    Args:
        predictions: pd.Series of predicted prices.
        actuals: pd.Series of actual prices.
        title: plot title.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(actuals.index, actuals.values, label="Actual", marker="o")
    plt.plot(predictions.index, predictions.values, label="Predicted", marker="o")
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_gbm_band(
    all_paths: np.ndarray,
    actual_prices: pd.Series = None,
    title: str = "GBM Simulated Paths with Confidence Band"
):
    """
    Plots GBM forecast band and actual prices if provided.

    Args:
        all_paths: np.ndarray of shape (n_paths, n_days)
        actual_prices: pd.Series of actual future prices (optional)
    """
    n_days = all_paths.shape[1]
    days = np.arange(1, n_days + 1)

    lower = np.percentile(all_paths, 10, axis=0)
    upper = np.percentile(all_paths, 90, axis=0)
    mean = all_paths.mean(axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(days, mean, label="Mean Forecast", color="orange")
    plt.fill_between(days, lower, upper, color="orange", alpha=0.2, label="10–90% Band")

    if actual_prices is not None:
        plt.plot(days, actual_prices.values, label="Actual Prices", color="blue", marker="o")

    plt.title(title)
    plt.xlabel("Days Ahead")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

