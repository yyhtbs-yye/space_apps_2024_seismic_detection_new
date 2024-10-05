# --------------------GPU Lib---------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EarthquakeDataset
from nets.localizer import FPN1DLocalizer as Localizer
from tools import mkdir
import psutil

def log_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / (1024 ** 3)} GB")  # RSS in GB

# ------------------------------------------------------------
# Device configuration
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 76
num_epochs = 200000
learning_rate = 0.0001

# Dataset and DataLoader
train_dataset = EarthquakeDataset(data_folder='data/lunar/training/downsample_data/S12_GradeA/',
                                  label_folder='data/lunar/training/labels/S12_GradeA/')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# Model
model = Localizer(num_pools=4, in_channels=1, mid_channels=12, kernel_size=13).to(device)

# Loss function and optimizer
# For classification with imbalance, using CrossEntropyLoss with class weights
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pth_path = "save/localizer/trial_2/"

mkdir(pth_path)

# Training loop
for epoch in range(num_epochs):

    log_memory_usage()  # Track memory at the start of each epoch

    print(f"Memory usage: {psutil.virtual_memory().percent}%")
    print(f"Number of threads: {psutil.Process().num_threads()}")

    model.train()  # Set the model to 4training mode
    running_loss = 0.0

    for i, (x, y, p) in enumerate(train_loader):    
        x = x.to(device)  # Send input to device (CPU/GPU)
        
        p = p.to(device)  # Send labels to device (CPU/GPU)

        # Forward pass
        p_hat = model(x)

        # Calculate loss
        loss = criterion(p_hat, p)
        
        # Backward and optimize
        optimizer.zero_grad()  # Zero the parameter gradients
        loss.backward()  # Compute gradients

        optimizer.step()  # Update parameters

        running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.9f}')

    print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {running_loss / len(train_loader):.9f}')

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'{pth_path}quake_localization_model_{epoch:05}.pth')



