import numpy as np

from lseg.preprocessing import load_price_data
from lseg.metrics import moving_average_baseline, evaluate_predictions


def main():
    df = load_price_data('./data/Price_History_2025.xlsx')

    # Baseline prediction
    predicted_prices = moving_average_baseline(df, window=5)

    # Evaluate performance
    metrics = evaluate_predictions(predicted_prices, df['Price'])
    print(metrics)


def output(prices):
    pass


if __name__ == '__main__':
    main()
