import os.path as osp
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import scipy.signal
from signal_fx import *
from seismic_fx import *

root = 'data'
planet = 'lunar'
dstype = 'training'
subaux = 'S12_GradeA'

# Construct the path to the metadata CSV file
data_folder_path = osp.join(root, planet, dstype, 'downsample_data', subaux)
label_folder_path = osp.join(root, planet, dstype, 'labels', subaux)

all_feature_df = pd.DataFrame()

def process_file(filename):
    # Skip non-csv files
    if ".csv" not in filename:
        return pd.DataFrame()

    print(filename)
    data_df = pd.read_csv(osp.join(data_folder_path, filename))
    label_df = pd.read_csv(osp.join(label_folder_path, filename))

    signal = data_df['velocity(m/s)']  # The signal column
    time = data_df['time_rel(sec)']  # The corresponding time column
    fs = 1 / (time[1] - time[0])  # Sampling frequency based on time difference

    signal = pd.Series(scipy.signal.wiener(signal))

    sta_lta_ratios, sta_out, lta_out = stalta_classic(time, signal, fs, sta_factor=50, lta_factor=2000)
    
    # Define a window size for the sliding window calculations (e.g., 100 samples)
    window_size = int(fs * 50)  # 10-second window as an example

    # Calculate sliding window features
    rms = sliding_window_rms(signal, window_size)
    energy = sliding_window_energy(signal, window_size)
    zcr = sliding_window_zero_crossing_rate(signal, window_size)
    teager_kaiser_energy = sliding_window_teager_kaiser_energy(signal, window_size)
    peak_to_peak = sliding_window_peak_to_peak(signal, window_size)
    shannon_entropy = sliding_window_shannon_entropy(signal, window_size)
    fractal_dimension = sliding_window_fractal_dimension(signal, window_size)
    label = label_df['label']

    # Create a new DataFrame for features
    feature_df = pd.DataFrame({
        'sta_lta_ratio': sta_lta_ratios,
        'sta': sta_out,
        'lta': lta_out,
        'rms': rms,
        'energy': energy,
        'zero_crossing_rate': zcr,
        'teager_kaiser_energy': teager_kaiser_energy,
        'peak_to_peak': peak_to_peak,
        'shannon_entropy': shannon_entropy,
        'fractal_dimension': fractal_dimension,
        'label': label,
    })

    return feature_df

# Use ProcessPoolExecutor to parallelize the loop
with ProcessPoolExecutor() as executor:
    # Get a list of all CSV files in the data folder
    filenames = [f for f in os.listdir(data_folder_path) if ".csv" in f]

    # Submit the process_file function to the executor for each filename
    results = list(tqdm(executor.map(process_file, filenames), total=len(filenames)))

# Concatenate all the feature DataFrames
all_feature_df = pd.concat(results, ignore_index=True)

# Save the final DataFrame to CSV
all_feature_df.to_csv("downsample_fx_parallel.csv", index=False)
