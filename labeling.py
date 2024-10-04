
import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt

from utils import onehot_labeling
from tqdm import tqdm

from tools import mkdir

# Define the root path, planet, dataset type, metadata filename, and auxiliary data folder
root = 'data'
planet = 'lunar'
dstype = 'training'
meta_filename = 'apollo12_catalog_GradeA_final'
subaux = 'S12_GradeA'


# Construct the path to the metadata CSV file
metafile_path = osp.join(root, planet, dstype, 'catalogs', meta_filename) + ".csv"

in_folder_path = osp.join(root, planet, dstype, 'downsample_data', subaux)

out_folder_path = osp.join(root, planet, dstype, 'labels', subaux)


# Read the metadata CSV file
df = pd.read_csv(metafile_path)

mkdir(out_folder_path)

# Iterate through each row in the metadata DataFrame
for i, row in tqdm(df.iterrows(), total=len(df)):

    filename = row['filename']
    filepath = osp.join(root, planet, dstype, 'downsample_data', subaux, filename) + ".csv"
    output_filepath = osp.join(out_folder_path, filename) + ".csv"

    # Check if the file exists before proceeding
    if osp.exists(filepath):
        print(filepath)

        # Check if the output file already exists
        if osp.exists(output_filepath):
            # If the output file exists, load the existing dataframe
            df_labeled = pd.read_csv(output_filepath)
        else: # Read the seismic data file
            df_labeled = pd.read_csv(filepath)
            df_labeled = df_labeled[['time_rel(sec)']]
            df_labeled['label'] = 0

        # Get the time relative from the metadata row
        time_rel = row['time_rel(sec)']

        df_labeled = onehot_labeling(df_labeled, 'time_rel(sec)', [time_rel], 'label')

        # Save the new DataFrame to a CSV file
        df_labeled.to_csv(output_filepath, index=False)
