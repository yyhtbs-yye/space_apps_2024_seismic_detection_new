import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class DynamicConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DynamicConv1D, self).__init__()
        
        # Conv layer to generate the dynamic convolution weights
        self.wx = nn.Conv1d(in_channels, out_channels * in_channels * kernel_size, 
                            kernel_size=kernel_size, 
                            padding=padding, stride=stride)
        
        # Store the hyperparameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def unfold1d(self, x):
        """
        Imitates im2col for 1D input by appending shifted versions of the input
        according to the kernel size, then stacking them to simulate sliding windows.
        """
        # Apply padding to the input for the convolution operation
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding))

        batch_size, channels, length = x.size()

        # List to hold shifted versions of the input
        unfolded = []

        # Collect shifted windows
        for i in range(self.kernel_size):
            unfolded.append(x[:, :, i:i + length - self.kernel_size + 1:self.stride])

        # Stack along a new dimension to mimic unfolding
        unfolded = torch.stack(unfolded, dim=2)  # Shape: (batch_size, channels, kernel_size, output_length)
        
        return unfolded

    def forward(self, x):
        batch_size, in_channels, seq_len = x.size()

        # Generate dynamic weights using input x through wx layer
        dynamic_weights = self.wx(x)  # Shape: (batch_size, out_channels * in_channels * kernel_size, seq_len)
        dynamic_weights = dynamic_weights.view(batch_size, self.out_channels, self.in_channels, self.kernel_size, -1)

        dynamic_weights = F.softmax(dynamic_weights / math.sqrt(self.kernel_size), dim=3)  # Normalize across kernel size

        # Unfold the input x into patches for im2col-like operation
        x_unfolded = self.unfold1d(x)  # Shape: (batch_size, in_channels, kernel_size, output_length)

        # Perform the inner product (im2col(x) @ w(x))
        y = torch.einsum('bckt,bockt->bot', [x_unfolded, dynamic_weights])

        return y

# Example usage of DynamicConv1D
if __name__ == "__main__":
    # Initialize the dynamic convolution layer
    in_channels = 5
    out_channels = 12
    kernel_size = 3
    dynamic_conv = DynamicConv1D(in_channels, out_channels, kernel_size=kernel_size, padding=1)

    # Example input data (batch_size=8, 1 channel, sequence_length=100)
    example_input = torch.randn(8, 5, 100)

    # Forward pass through the custom layer
    output = dynamic_conv(example_input)

    print("Output shape:", output.shape)  # Expected shape: (8, 16, 100) depending on stride and padding
