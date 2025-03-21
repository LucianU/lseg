import numpy as np
import pandas as pd

def load_price_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the index price data from an Excel file.

    Args:
        filepath: Path to the .xlsx file.

    Returns:
        DataFrame with columns: ['Date', 'Price', 'LogReturn']
    """
    df = pd.read_excel(filepath)

    # Standardize column names
    df.columns = df.columns.str.strip()
    df.rename(columns={
        'Exchange Date': 'Date',
        'Index Price': 'Price'
    }, inplace=True)

    # Convert to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Optionally drop missing or zero prices
    df = df[df['Price'].notnull() & (df['Price'] > 0)]

    # Compute log returns (used in models like GBM)
    df['LogReturn'] = df['Price'].apply(lambda x: float(x))
    df['LogReturn'] = df['LogReturn'].apply(float).pct_change()
    df['LogReturn'] = df['LogReturn'].apply(lambda r: pd.NA if r <= -1 else r)  # exclude huge errors
    df['LogReturn'] = np.log(df['Price'] / df['Price'].shift(1))

    return df.dropna().reset_index(drop=True)


def add_moving_average(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Adds a moving average column to the DataFrame.
    """
    df[f'MA_{window}'] = df['Price'].rolling(window=window).mean()
    return df

