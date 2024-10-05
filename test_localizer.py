# --------------------GPU Lib---------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EarthquakeDataset
# from nets.localizer import FPN1DLocalizer as Localizer
from nets.localizer import SimpleCNNLocalizer as Localizer

import psutil
def log_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / (1024 ** 3)} GB")  # RSS in GB

# ------------------------------------------------------------
# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 76
num_epochs = 2000
learning_rate = 0.0001

# Dataset and DataLoader
test_dataset = EarthquakeDataset(data_folder='data/lunar/test/downsample_data/S12_GradeB/',
                                  label_folder=None, is_testing=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = Localizer(num_layers=7, in_channels=1, mid_channels=8, kernel_size=13).to(device)

# Load the saved model parameters
model.load_state_dict(torch.load('save/localizer/trial_0/quake_localization_model_01620.pth'))


batch = next(iter(test_loader))

pred = model(batch[0].to(device)).flatten()

a = 1