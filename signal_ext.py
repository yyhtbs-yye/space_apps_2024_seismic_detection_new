import librosa
import numpy as np

def lpc(signal, order):
    """
    Compute LPC coefficients using librosa's lpc function.
    
    Parameters:
    signal : array-like
        The input signal from which to calculate LPC coefficients.
    order : int
        The order of the LPC (number of coefficients to return).
    
    Returns:
    lpc_coeffs : ndarray
        LPC coefficients (including the leading 1 term).
    """
    return librosa.lpc(signal, order)


if __name__=="__main__":
    # Example: Generate a sample seismic signal (or use your own data)
    fs = 100  # Sampling rate in Hz
    t = np.linspace(0, 1, fs)
    signal = 0.7 * np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 15 * t)

    # Apply LPC with librosa
    order = 10  # LPC order
    lpc_coeffs = lpc(signal, order)

    # Output LPC coefficients
    print("LPC Coefficients:", lpc_coeffs)
