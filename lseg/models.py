import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq
from statsmodels.tsa.arima.model import ARIMA


def simulate_gbm_paths(
    df: pd.DataFrame,
    n_days: int = 5,
    n_paths: int = 1000,
    seed: int = 42
) -> tuple[pd.Series, np.ndarray]:
    """
    Simulates GBM future prices based on historical log return statistics.

    Returns:
        (mean_forecast, all_paths):
            - mean_forecast: average price path over n_days
            - all_paths: raw simulation array of shape (n_paths, n_days)
    """
    np.random.seed(seed)

    last_price = df['Price'].iloc[-1]
    mu = df['LogReturn'].mean()
    sigma = df['LogReturn'].std()
    dt = 1

    random_shocks = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_days))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * random_shocks
    log_paths = np.cumsum(increments, axis=1)
    simulated_paths = last_price * np.exp(log_paths)

    mean_forecast = simulated_paths.mean(axis=0)

    return pd.Series(mean_forecast), simulated_paths


def fft_predict(
    prices: pd.Series,
    n_future: int = 5,
    n_freq: int = 5
) -> pd.Series:
    """
    Predicts future prices using FFT by extrapolating dominant frequency components.

    Args:
        prices: pd.Series of historical prices.
        n_future: Number of days to predict.
        n_freq: Number of dominant frequencies to keep.

    Returns:
        pd.Series of predicted prices (length = n_future).
    """
    N = len(prices)
    prices_centered = prices - prices.mean()

    # FFT
    freqs = fftfreq(N)
    fft_values = fft(prices_centered.to_numpy())

    # Identify top frequencies
    magnitude = np.abs(fft_values[:N // 2])
    top_indices = np.argsort(magnitude)[-n_freq:]

    # Keep only dominant frequencies
    filtered_fft = np.zeros_like(fft_values)
    for idx in top_indices:
        filtered_fft[idx] = fft_values[idx]
        filtered_fft[-idx] = fft_values[-idx]  # for symmetry

    # Inverse FFT for reconstruction
    reconstructed = np.real(ifft(filtered_fft)) + prices.mean()

    # Fit a simple extrapolation (continue the signal forward)
    delta = reconstructed[-1] - reconstructed[-2]  # last step
    predicted = [reconstructed[-1] + i * delta for i in range(1, n_future + 1)]

    return pd.Series(predicted)


def arima_predict(
    prices: pd.Series,
    n_future: int = 5,
    order: tuple = (3, 1, 2)
) -> pd.Series:
    """
    Predicts future prices using a fixed ARIMA model.

    Args:
        prices: Historical price series (1D).
        n_future: How many steps to forecast.
        order: ARIMA order (p, d, q), default from auto_arima result.

    Returns:
        pd.Series of predicted prices.
    """
    model = ARIMA(prices, order=order)
    arima_model = model.fit()
    forecast = arima_model.forecast(steps=n_future)
    return pd.Series(forecast.values)

