import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class EarthquakeDataset(Dataset):
    def __init__(self, data_folder, label_folder, is_testing=False):
        self.data_folder = data_folder
        self.label_folder = label_folder
        self.csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        self.max_length = -1
        self.data = []  

        # Determine the maximum sequence length first
        for csv_file in self.csv_files:
            data_csv_path = os.path.join(self.data_folder, csv_file)
            df = pd.read_csv(data_csv_path)
            length = len(df['velocity(m/s)'].values)
            if length > self.max_length:
                self.max_length = length

        # Preload and pad all CSV data
        for csv_file in self.csv_files:
            data_csv_path = os.path.join(self.data_folder, csv_file)
            data_df = pd.read_csv(data_csv_path)

            # Get velocity and label as tensors
            x = torch.tensor(data_df['velocity(m/s)'].values, dtype=torch.float32)
            # Pad the velocity (x) and label (y) sequences to match the maximum length
            padded_x = torch.zeros(self.max_length, dtype=torch.float32)
            padded_x[:x.size(0)] = x * 1e10  # Scaling factor

            if not is_testing:
                label_csv_path = os.path.join(self.label_folder, csv_file)
                label_df = pd.read_csv(label_csv_path)
                y = torch.tensor(label_df['label'].values, dtype=torch.float32)
                padded_y = torch.zeros(self.max_length, dtype=torch.float32)
                padded_y[:y.size(0)] = y
                # Calculate timestamp percentage
                timestamp_idx = torch.nonzero(y == 1).item()  # Assuming only one "1" exists in y
                timestamp_percentage = timestamp_idx / x.size(0)  # Percentage of sequence
                timestamp_percentage = torch.tensor([timestamp_percentage], dtype=torch.float32)
                padded_y = padded_y.long()

            else:
                padded_y = []
                timestamp_percentage = []

            if is_testing:
                # Store padded data and timestamp percentage
                self.data.append((padded_x.view(1, -1), padded_y, timestamp_percentage, csv_file))
            else:
                # Store padded data and timestamp percentage
                self.data.append((padded_x.view(1, -1), padded_y, timestamp_percentage))

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        # Return preloaded, padded data and timestamp percentage
        return self.data[idx]

if __name__ == "__main__":  # Test Code
    # Assuming all your CSV files are stored in the folder 'downsampled_signals_and_sampels/'
    data_folder = 'downsampled_signals_and_sampels/'
    label_folder = 'downsampled_signals_and_sampels/'
    dataset = EarthquakeDataset(data_folder, label_folder)

    # Create a DataLoader to load the data in batches
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Example of iterating through the dataset
    for x, y, timestamp_percentage in dataloader:
        print("Input (x):", x.shape)
        print("Target (y):", y.shape)
        print("Timestamp Percentage:", timestamp_percentage)
