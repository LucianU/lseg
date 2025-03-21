import numpy as np
import pandas as pd

from lseg.preprocessing import load_price_data
from lseg.models import simulate_gbm_paths, fft_predict, arima_predict
from lseg.metrics import evaluate_predictions
from lseg.plot_utils import plot_predictions, plot_gbm_band


def main():
    df = load_price_data('./data/Price_History_2025.xlsx')

    evaluate_gbm(df)
    evaluate_arima(df)
    evaluate_baseline(df)
    evaluate_fft(df, plot=True)
    #show_gbm_band(df)


def evaluate_baseline(df, window=5, n_future=5, plot=False):
    # Step 1: Split
    train_df = df.iloc[:-n_future].copy()
    true_future = df['Price'].iloc[-n_future:].reset_index(drop=True)

    # Step 2: Compute last MA from train data
    last_ma = train_df['Price'].rolling(window=window).mean().iloc[-1]

    # Step 3: Predict the same MA value for all future days (naive baseline)
    predicted_prices = pd.Series([last_ma] * n_future)

    # Step 4: Evaluate
    metrics = evaluate_predictions(predicted_prices, true_future)
    print("Baseline Evaluation Metrics:", metrics)

    # Step 5: Plot
    if plot:
        plot_predictions(predicted_prices, true_future, title="Baseline: Moving Average Prediction vs Actual")

def evaluate_gbm(df, plot=False):
    # Step 1: Choose split point
    n_future = 5
    train_df = df.iloc[:-n_future].copy()
    true_future = df['Price'].iloc[-n_future:].reset_index(drop=True)

    # Step 2: Simulate GBM predictions using training data only
    predicted_prices, _ = simulate_gbm_paths(train_df, n_days=n_future)

    # Step 3: Evaluate
    metrics = evaluate_predictions(predicted_prices, true_future)
    print("GBM Evaluation Metrics:", metrics)

    # Step 4: Plot
    if plot:
        plot_predictions(predicted_prices, true_future, title="GBM: Prediction vs Actual")

def evaluate_arima(df, plot=False):
    n_future = 5
    train_df = df.iloc[:-n_future].copy()
    true_future = df['Price'].iloc[-n_future:].reset_index(drop=True)

    # Run ARIMA prediction
    arima_preds = arima_predict(train_df['Price'], n_future=n_future)

    # Evaluate and plot
    metrics = evaluate_predictions(arima_preds, true_future)
    print("ARIMA Evaluation Metrics:", metrics)

    if plot:
        plot_predictions(arima_preds, true_future, title="ARIMA: Prediction vs Actual")

def evaluate_fft(df, plot=False):
    # Step 1: Choose split point
    n_future = 5
    train_df = df.iloc[:-n_future].copy()
    true_future = df['Price'].iloc[-n_future:].reset_index(drop=True)

    # Step 2: Predict with FFT
    fft_preds = fft_predict(train_df['Price'], n_future=n_future)

    # Step 3: Evaluate
    metrics = evaluate_predictions(fft_preds, true_future)
    print("FFT Evaluation Metrics:", metrics)

    # Step 4: Plot
    if plot:
        plot_predictions(fft_preds, true_future, title="FFT-Based Prediction vs Actual")

def show_gbm_band(df):
    n_future = 5
    train_df = df.iloc[:-n_future].copy()
    true_future = df['Price'].iloc[-n_future:].reset_index(drop=True)

    # Simulate GBM
    _, all_paths = simulate_gbm_paths(train_df, n_days=n_future)

    # Plot band vs actual
    plot_gbm_band(all_paths, actual_prices=true_future)


def output(prices):
    pass


if __name__ == '__main__':
    main()
