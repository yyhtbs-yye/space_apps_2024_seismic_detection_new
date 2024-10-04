import os.path as osp
import os

import pywt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
from tools import mkdir
from matplotlib.gridspec import GridSpec

root = 'data'
planet = 'lunar'
dstype = 'training'
subaux = 'S12_GradeA'

# Define paths
in_folder_path = osp.join(root, planet, dstype, 'downsample_data', subaux)
aux_folder_path = osp.join(root, planet, dstype, 'labels', subaux)
out_folder_path = osp.join(root, planet, dstype, 'wavelet_transform', subaux)

mkdir(out_folder_path)

# Loop through all files in the folder
for filename in tqdm(os.listdir(in_folder_path)):
    
    if ".csv" not in filename:
        continue

    # Load the data from the CSV file
    in_file_path = osp.join(in_folder_path, filename)
    df = pd.read_csv(in_file_path)

    aux_file_path = osp.join(aux_folder_path, filename)
    labeled_df = pd.read_csv(aux_file_path) # labeled_df has two columns, the 'time_rel(sec)' and 'label'

    # Perform Continuous Wavelet Transform (CWT)
    scales = np.arange(1, 64)  # Define scale range
    coefficients, frequencies = pywt.cwt(df['velocity(m/s)'], scales, 'morl')  # Using Morlet wavelet

    # Create a 3x1 subplot layout where the first two subplots are combined
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 1, height_ratios=[2, 1])  # First two subplots are larger
    
    # First plot (Morlet Transform, part 1 of 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.log(np.abs(coefficients)), extent=[df['time_rel(sec)'].min(), df['time_rel(sec)'].max(), scales.min(), scales.max()],
            cmap='viridis', vmax=-20, vmin=-25, aspect='auto', origin='lower')
    ax1.set_ylabel('Scale')
    ax1.set_title(f'Wavelet Transform (Morlet) of Velocity Signal - {filename}')

    # Twin axis to plot the label on top of the first plot
    ax1b = ax1.twinx()
    ax1b.plot(labeled_df['time_rel(sec)'], labeled_df['label'], color='red', linewidth=0.75, label='Label')
    ax1b.set_ylabel('Label')
    ax1b.set_ylim(-0.1, 1.1)  # Ensure the y-axis is appropriate for binary labels (0 or 1)
    ax1b.tick_params(axis='y', colors='red')  # Color the ticks to differentiate
    ax1b.legend(loc='upper right')  # Optional: add a legend for the label

    # Third plot (Original Signal)
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Share x-axis with ax1
    ax3.plot(df['time_rel(sec)'], df['velocity(m/s)'], color='blue', linewidth=0.5)
    ax3.set_title('Original Signal (Velocity)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_xlabel('Time (sec)')

    # Optionally, you can hide the x-ticks of the first plot (ax1) to avoid clutter
    ax1.tick_params(axis='x', which='both', labelbottom=False)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure as PNG in the output folder
    output_png_path = osp.join(out_folder_path, filename.replace('.csv', '.png'))
    plt.savefig(output_png_path)
    plt.close(fig)  # Close the figure after saving to prevent display
