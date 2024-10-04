# --------------------GPU Lib---------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EarthquakeDataset
# from nets.localizer import FPN1DLocalizer as Localizer
from nets.localizer import SimpleCNNLocalizer as Localizer

# ------------------------------------------------------------
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 76
num_epochs = 2000
learning_rate = 0.0001

# Dataset and DataLoader
train_dataset = EarthquakeDataset(csv_folder='downsampled_signals_and_sampels/S12_GradeA/')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = Localizer().to(device)

# Loss function and optimizer
# For classification with imbalance, using CrossEntropyLoss with class weights
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
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

        # for param in model.parameters():
        #     if param.grad is not None:
        #         print(param.grad.norm())

        optimizer.step()  # Update parameters
        

        running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.9f}')

    print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {running_loss / len(train_loader):.9f}')


# Save the trained model
torch.save(model.state_dict(), 'quake_detection_model.pth')
