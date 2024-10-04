import numpy as np
import pandas as pd

def stalta_classic(t, signal, fs, sta_factor=5, lta_factor=30, is_centred=False):
    """
    Function to perform STA/LTA detection.

    Parameters:
    t (pandas dataframe): Time array corresponding to the signal.
    signal (pandas dataframe): The input signal to analyze.
    fs (float): The sampling frequency of the signal.
    sta_factor (float): Scaling factor to adjust the STA length (in seconds).
    lta_factor (float): Scaling factor to adjust the LTA length (in seconds).

    Returns:
    sta_lta_ratios (numpy array): The STA/LTA ratio over time.
    sta_out (numpy array): The STA values over time.
    lta_out (numpy array): The LTA values over time.
    """

    # Convert STA and LTA factors to sample lengths
    sta_window = int(sta_factor * fs)
    lta_window = int(lta_factor * fs)
    
    # Short-term average (STA) and Long-term average (LTA) calculation
    if is_centred:
        # Centered rolling window
        sta_out = signal.abs().rolling(window=sta_window, center=True, min_periods=1).mean()
        lta_out = signal.abs().rolling(window=lta_window, center=True, min_periods=1).mean()
    else:
        # Trailing rolling window (no future data)
        sta_out = signal.abs().rolling(window=sta_window, min_periods=1).mean()
        lta_out = signal.abs().rolling(window=lta_window, min_periods=1).mean()

    # STA/LTA ratio calculation
    sta_lta_ratios = sta_out / lta_out.replace(0, np.nan)  # Prevent division by zero by replacing 0s with NaN
    sta_lta_ratios = sta_lta_ratios.fillna(0)  # Replace NaNs from division by zero with 0
    
    return sta_lta_ratios, sta_out, lta_out

