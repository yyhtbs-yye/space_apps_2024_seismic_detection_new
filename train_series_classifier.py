# --------------------GPU Lib---------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EarthquakeDataset
from model import QuakeDetectionNet
# --------------------CPU Lib---------------------------------
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# --------------------UTILS Lib---------------------------------
from utils import calculate_confusion_matrix
# ------------------------------------------------------------
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 76
num_epochs = 2000
learning_rate = 0.01

# Dataset and DataLoader
train_dataset = EarthquakeDataset(data_folder='data/lunar/training/downsample_data/S12_GradeA/',
                                  label_folder='data/lunar/training/label/S12_GradeA/')
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

initial_sta_length, initial_lta_length = 60, 200

# Model
model = QuakeDetectionNet(initial_sta_length, initial_lta_length).to(device)

# Handling class imbalance (if it's a classification problem)
# Calculate class weights if there is class imbalance
def compute_class_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

# Assuming y is a list or array of labels
y_labels = []
for _, y, _ in train_dataset:
    y_labels.extend(y.tolist())  # Assuming y is a tensor
class_weights = compute_class_weights(y_labels)

# Loss function and optimizer
# For classification with imbalance, using CrossEntropyLoss with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0  # Initialize totals for each epoch

    for i, (x, y, p) in enumerate(train_loader):
        x = x.to(device)  # Send input to device (CPU/GPU)
        y = y.to(device)  # Send label to device (CPU/GPU)

        # Forward pass
        outputs, slta = model(x)

        # Calculate loss
        loss = criterion(outputs.view(-1, 2), y.view(-1))
        
        # Backward and optimize
        optimizer.zero_grad()  # Zero the parameter gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        
        running_loss += loss.item()
        # Get predicted labels
        predicted_labels = torch.argmax(outputs, dim=-1).view(-1)  # Shape [B*T]
        true_labels = y.view(-1)  # Shape [B*T]

        # Calculate confusion matrix for the current batch
        TP, FP, TN, FN = calculate_confusion_matrix(predicted_labels, true_labels)

        # Accumulate values for epoch-level statistics
        total_TP += TP
        total_FP += FP
        total_TN += TN
        total_FN += FN

        # if (i+1) % 10 == 0:  # Print every 10 batches
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.9f}')

    # Calculate metrics based on the confusion matrix
    accuracy = (total_TP + total_TN) / (total_TP + total_FP + total_TN + total_FN)
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {running_loss / len(train_loader):.9f}')
    print(f'Confusion Matrix: TP: {total_TP}, FP: {total_FP}, TN: {total_TN}, FN: {total_FN}')
    print(f'Accuracy: {accuracy:.9f}, Precision: {precision:.9f}, Recall: {recall:.9f}, F1 Score: {f1_score:.9f}')


    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'quake_detection_model_{epoch:05}.pth')

    print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {running_loss / len(train_loader):.9f}')

