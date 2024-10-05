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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 76
num_epochs = 2000
learning_rate = 0.0001

# Dataset and DataLoader
train_dataset = EarthquakeDataset(csv_folder='data/lunar/test/downsample_data/S12_GradeB/', is_testing=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# Model
model = Localizer().to(device)

# Load the saved model parameters
model.load_state_dict(torch.load('quake_localization_model.pth'))


batch = next(iter(train_loader))

model(batch[0])

a = 1