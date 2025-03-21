import numpy as np
import pandas as pd


def moving_average_baseline(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Predicts next-day price as today's moving average.

    Returns a shifted Series of predictions aligned with actual values.
    """
    ma = df['Price'].rolling(window=window).mean()
    prediction = ma.shift(1)  # Predict tomorrow's price using today's MA
    return prediction


def evaluate_predictions(predictions: pd.Series, actuals: pd.Series) -> dict:
    """
    Evaluates prediction performance.

    Args:
        predictions: Predicted price series.
        actuals: Actual price series.

    Returns:
        Dictionary with MAE, RMSE, and directional accuracy.
    """
    # Drop rows with NaNs in either
    mask = predictions.notna() & actuals.notna()
    predictions = predictions[mask]
    actuals = actuals[mask]

    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    direction_acc = np.mean(np.sign(predictions.diff()) == np.sign(actuals.diff()))

    return {
        'MAE': mae,
        'RMSE': rmse,
        'Directional Accuracy': direction_acc
    }

