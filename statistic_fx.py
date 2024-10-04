import numpy as np

def sliding_window_std(signal, window_size):
    """
    Calculate the standard deviation of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.Series: The standard deviation of the signal for each window.
    """
    return signal.rolling(window=window_size, min_periods=1).std()

def sliding_window_skewness(signal, window_size):
    """
    Calculate the skewness of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.Series: The skewness of the signal for each window.
    """
    return signal.rolling(window=window_size, min_periods=1).skew()

def sliding_window_kurtosis(signal, window_size):
    """
    Calculate the kurtosis of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.Series: The kurtosis of the signal for each window.
    """
    return signal.rolling(window=window_size, min_periods=1).kurt()

