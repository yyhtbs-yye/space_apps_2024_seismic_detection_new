import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Gaussian window kernel with learnable standard deviation (std)
class LearnableGaussianWindow(nn.Module):
    def __init__(self, initial_window_length, min_std=1e-2):
        super(LearnableGaussianWindow, self).__init__()
        
        # Learnable parameter: window spread, controls the effective window size
        self.window_size_param = nn.Parameter(torch.tensor(float(initial_window_length)))
        self.min_std = min_std  # Minimum std deviation for stability

    def forward(self, length):
        t = torch.linspace(-1.0, 1.0, length, device=self.window_size_param.device)
        
        # Use the learned window size parameter as the standard deviation for the Gaussian window
        std_dev = torch.abs(self.window_size_param) + self.min_std  # Ensure std_dev is positive
        window = torch.exp(-0.5 * (t / std_dev) ** 2)
        return window / window.sum()  # Normalize the window

# STA and LTA layer with learnable window size
class SLTALayer(nn.Module):
    def __init__(self, initial_sta_length, initial_lta_length):
        super(SLTALayer, self).__init__()

        # Learnable Gaussian windows for STA and LTA
        self.sta_window = LearnableGaussianWindow(initial_sta_length)
        self.lta_window = LearnableGaussianWindow(initial_lta_length)

    def forward(self, x):
        B_, C_, T_ = x.size()

        # Compute the window lengths based on the input sequence length
        sta_window_length = min(T_, 50)  # You can dynamically change this if needed
        lta_window_length = min(T_, 200) # You can dynamically change this if needed
        
        # Get the learned windows (differentiate w.r.t. window size)
        sta_window = self.sta_window(sta_window_length).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, sta_window_length)
        lta_window = self.lta_window(lta_window_length).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, lta_window_length)

        # Apply 1D convolution with learned windows
        sta = F.conv1d(x, sta_window, padding='same')  # STA over short-term window
        lta = F.conv1d(x, lta_window, padding='same')  # LTA over long-term window

        # Compute STA/LTA ratio
        sta_lta_ratio = sta / (lta + 1e-6)  # Avoid division by zero

        return sta, lta, sta_lta_ratio

# Thresholding layer with a learnable threshold
class ThresholdLayer(nn.Module):
    def __init__(self, init_threshold=3.0):
        super(ThresholdLayer, self).__init__()
        # Learnable threshold parameter
        self.threshold = nn.Parameter(torch.tensor(init_threshold))

    def forward(self, sta_lta_ratio):
        # Sigmoid function applied to STA/LTA ratio and learned threshold
        return torch.sigmoid(sta_lta_ratio - self.threshold)

# Full Quake Detection Network
class QuakeDetectionNet(nn.Module):
    def __init__(self, initial_sta_length, initial_lta_length):
        super(QuakeDetectionNet, self).__init__()

        # STA and LTA layers with learnable window sizes
        self.sta_lta_layer = SLTALayer(initial_sta_length, initial_lta_length)

        # A simple fully connected layer to make a detection decision
        self.fc = nn.Linear(3, 2)  # Assuming input length matches window length

    def forward(self, x):

        x = x.unsqueeze(1)
        # x is assumed to be of shape (B_, 1, T_)

        # Get STA, LTA, and STA/LTA ratio from the learnable window layers
        sta, lta, sta_lta_ratio = self.sta_lta_layer(x)

        # Apply the learnable threshold
        detections = torch.concat((sta, lta, sta_lta_ratio), dim=1)
        output = self.fc(detections.permute(0, 2, 1)).squeeze()

        return output, detections

# Example of using the model
if __name__ == "__main__":
    # Initialize the model
    initial_sta_length = 50
    initial_lta_length = 200
    model = QuakeDetectionNet(initial_sta_length, initial_lta_length)

    # Example input data (B_=8, 1 channel, T_=1000)
    example_input = torch.randn(8,1000)

    # Forward pass
    output, detections = model(example_input)

    print("Model Output (Quake Detection):", output)
    print("STA/LTA Detections:", detections)
