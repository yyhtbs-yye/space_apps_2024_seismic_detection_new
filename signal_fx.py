import numpy as np
import pandas as pd 
from scipy.signal import welch
from scipy.stats import entropy
from scipy.signal import find_peaks

def sliding_window_rms(signal, window_size):
    """
    Calculate the RMS (Root Mean Square) of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.Series: The RMS of the signal for each window.
    """
    return signal.rolling(window=window_size, min_periods=1).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)


def sliding_window_energy(signal, window_size):
    """
    Calculate the energy (sum of squared values) of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.Series: The energy of the signal for each window.
    """
    return signal.rolling(window=window_size, min_periods=1).apply(lambda x: np.sum(x**2), raw=True)

def sliding_window_zero_crossing_rate(signal, window_size):
    """
    Calculate the zero-crossing rate of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.Series: The zero-crossing rate for each window.
    """
    return signal.rolling(window=window_size, min_periods=1).apply(lambda x: ((x[:-1] * x[1:]) < 0).sum(), raw=True)

def sliding_window_spectral_entropy(signal, window_size, fs=1.0):
    """
    Calculate the spectral entropy of the signal within a sliding window.
    
    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.
    fs (float): Sampling frequency of the signal.

    Returns:
    pd.Series: The spectral entropy of the signal for each window.
    """
    def compute_spectral_entropy(window):
        freqs, psd = welch(window, fs=fs)
        psd_norm = psd / psd.sum()  # Normalize the Power Spectral Density (PSD)
        return entropy(psd_norm)
    
    return signal.rolling(window=window_size, min_periods=1).apply(compute_spectral_entropy, raw=True)

def sliding_window_dominant_frequency(signal, window_size, fs=1.0):
    """
    Calculate the dominant frequency of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.
    fs (float): Sampling frequency of the signal.

    Returns:
    pd.Series: The dominant frequency of the signal for each window.
    """
    def compute_dominant_frequency(window):
        freqs, psd = welch(window, fs=fs)
        dominant_freq = freqs[np.argmax(psd)]  # Find frequency with maximum power
        return dominant_freq
    
    return signal.rolling(window=window_size, min_periods=1).apply(compute_dominant_frequency, raw=True)

def sliding_window_hjorth_parameters(signal, window_size):
    """
    Calculate Hjorth parameters (activity, mobility, complexity) within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.DataFrame: The Hjorth parameters (activity, mobility, complexity) for each window.
    """
    def compute_hjorth(window):
        activity = np.var(window)
        mobility = np.sqrt(np.var(np.diff(window)) / activity)
        complexity = (np.sqrt(np.var(np.diff(np.diff(window))) / np.var(np.diff(window))) / mobility)
        return pd.Series([activity, mobility, complexity], index=['activity', 'mobility', 'complexity'])
    
    return signal.rolling(window=window_size, min_periods=1).apply(lambda x: compute_hjorth(x), raw=False)

def sliding_window_teager_kaiser_energy(signal, window_size):
    """
    Calculate the Teager-Kaiser energy of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.Series: The Teager-Kaiser energy for each window.
    """
    def teager_kaiser(window):
        return np.mean(window[1:-1]**2 - window[:-2] * window[2:])
    
    return signal.rolling(window=window_size, min_periods=3).apply(teager_kaiser, raw=True)

def sliding_window_autocorrelation(signal, window_size, lag=1):
    """
    Calculate the autocorrelation of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.
    lag (int): The lag for autocorrelation calculation.

    Returns:
    pd.Series: The autocorrelation for each window.
    """
    def compute_autocorrelation(window):
        return window.autocorr(lag=lag)
    
    return signal.rolling(window=window_size, min_periods=lag+1).apply(compute_autocorrelation, raw=False)

def sliding_window_peak_to_peak(signal, window_size):
    """
    Calculate the peak-to-peak amplitude of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.Series: The peak-to-peak amplitude for each window.
    """
    return signal.rolling(window=window_size, min_periods=1).apply(lambda x: x.max() - x.min(), raw=True)

def sliding_window_shannon_entropy(signal, window_size):
    """
    Calculate the Shannon entropy of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.Series: The Shannon entropy for each window.
    """
    def compute_shannon_entropy(window):
        probabilities, _ = np.histogram(window, bins=10, density=True)
        probabilities = probabilities[probabilities > 0]  # Remove zero entries
        return -np.sum(probabilities * np.log2(probabilities))
    
    return signal.rolling(window=window_size, min_periods=1).apply(compute_shannon_entropy, raw=True)

def sliding_window_fractal_dimension(signal, window_size):
    """
    Calculate the fractal dimension of the signal within a sliding window.

    Parameters:
    signal (pd.DataFrame): The signal to analyze.
    window_size (int): The size of the sliding window.

    Returns:
    pd.Series: The fractal dimension for each window.
    """
    def compute_fractal_dimension(window):
        N = len(window)
        x = np.arange(N)
        return np.log(np.abs(np.diff(window)).sum()) / np.log(N)
    
    return signal.rolling(window=window_size, min_periods=2).apply(compute_fractal_dimension, raw=True)
