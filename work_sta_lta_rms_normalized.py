import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal as signal
from dsp import apply_bandpass_filter

from visualization import plot_sta_lta_ratio
from signal_fx import sliding_window_rms

root = 'data'
planet = 'lunar'
dstype = 'test'
subaux = 'S16_GradeB'
filename = 'xa.s16.00.mhz.1973-12-18HR00_evid00487.csv'

in_file_path = osp.join(root, planet, dstype, 'data', subaux, filename)

df = pd.read_csv(in_file_path)

minfreq = 0.5  # Minimum frequency (Hz)
maxfreq = 1.0  # Maximum frequency (Hz)
fs = 1 / (df['time_rel(sec)'][1] - df['time_rel(sec)'][0])  # Original sample rate (Assumed 10 Hz, adjust accordingly)

df['velocity(m/s)'] = apply_bandpass_filter(df['velocity(m/s)'], minfreq, maxfreq, fs)
df['velocity(m/s)'] = signal.wiener(df['velocity(m/s)'])

from seismic_fx import stalta_classic

t = df['time_rel(sec)']
x = df['velocity(m/s)']

# Perform STA/LTA detection, do not centre the signal. 
sta_lta_ratios, sta, lta = stalta_classic(df['time_rel(sec)'], df['velocity(m/s)'], fs, sta_factor=50, lta_factor=2000)

# Calculate the RMS envelope
rms_env = sliding_window_rms(x, 500)

plot_sta_lta_ratio(t, x, sta, lta, sta_lta_ratios * rms_env)


a = 0