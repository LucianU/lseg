import numpy as np


class Model:
    def __init__(self, data):
        self.data = data

    def predict(self):
        pass


def compute_moving_average(prices, window=5):
    return prices.rolling(window=window).mean()

def evaluate_model(predictions, actuals):
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    direction_acc = np.mean(np.sign(predictions.diff()) == np.sign(actuals.diff()))
    return {'MAE': mae, 'RMSE': rmse, 'Directional Accuracy': direction_acc}

def simulate_gbm():
    pass

def fft_filter_and_predict():
    pass

def plot_results():
    pass

def output(prices):
    pass


