import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

apple_data = '/Users/lucian/Documents/StockData/AAPL-5y.csv'

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime
    df = df.sort_values(by='Date')  # Ensure data is sorted by date
    df['Close/Last'] = df['Close/Last'].str.replace('$', '').astype(float)  # Convert Close price to float
    return df

def extract_timeseries(df):
    prices = df['Close/Last'].values
    N = len(prices)  # Number of data points
    time = np.arange(N)  # Simulated time index
    return time, prices

def get_price_frequencies(prices):
    N = len(prices)  # Number of data points
    freqs = fftfreq(N)  # Frequency components
    fft_values = fft(prices)  # Compute Fourier Transform

    # Only keep positive frequencies (FFT is symmetric)
    positive_freqs = freqs[:N // 2]
    positive_fft_values = np.abs(fft_values[:N // 2])

    return (positive_freqs, positive_fft_values)

def plot_price(t, price):
# Plot the original price data
    plt.figure(figsize=(10, 4))
    plt.plot(t, price, label="Stock Closing Prices", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Price over Time")
    plt.legend()
    plt.grid()
    plt.show()

def plot_frequency_spectrum(freqs, fft_values):
    # Plot the frequency spectrum (Fourier Transform)
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_values, label="FFT Magnitude", color="red")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Fourier Transform of Price Data")
    plt.legend()
    plt.grid()
    plt.show()

def analyze_aapl():
    df = load_data(apple_data)
    df = preprocess_data(df)
    t, prices = extract_timeseries(df)
    freqs, fft_values = get_price_frequencies(prices)

    #plot_price(t, prices)
    plot_frequency_spectrum(freqs, fft_values)


analyze_aapl()


def simulate_data():
    # Generate synthetic price data with a mix of trends and oscillations
    T = 200  # Number of time steps
    t = np.linspace(0, 10, T)  # Time vector

    # Simulated price = trend + oscillations + noise
    simulated_price = (
        50 + 2 * t  # Upward trend
        + 5 * np.sin(2 * np.pi * 0.2 * t)  # Low-frequency oscillation
        + 2 * np.sin(2 * np.pi * 1.5 * t)  # High-frequency oscillation
        + np.random.normal(0, 1, T)  # Noise
    )
    return t, simulated_price

