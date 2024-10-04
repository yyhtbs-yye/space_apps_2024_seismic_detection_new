import torch
import torch.nn as nn
import torch.optim as optim

class FPN1DLocalizer(nn.Module):
    def __init__(self, num_pools=10, features=256, in_channels=1, mid_channels=64, kernel_size=31):
        super(FPN1DLocalizer, self).__init__()
        
        self.num_pools = num_pools
        
        # Encoder: Each level has a Conv -> ReLU -> Conv -> ReLU block
        self.encoder_blocks = nn.ModuleList()
        self.encoder_blocks.append(nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2))
        for _ in range(num_pools):
            self.encoder_blocks.append(nn.Sequential(
                nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True)
            ))
            in_channels = mid_channels

        # Max pooling layers for downsampling
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        # Bottleneck layer
        bottleneck_out_channels = in_channels
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_out_channels, bottleneck_out_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck_out_channels * 2, bottleneck_out_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Feature Pyramid Network (FPN) layers
        self.fpn_blocks = nn.ModuleList()
        for i in range(num_pools):
            self.fpn_blocks.append(nn.Conv1d(mid_channels, mid_channels, kernel_size=1))  # 1x1 Conv for feature fusion
        
        # Spatial Pyramid Pooling (SPP) layers
        self.spp = nn.ModuleList([
            nn.AdaptiveMaxPool1d(output_size=output_size) for output_size in [1, 4, 8, 16]  # Pyramid with different scales
        ])

        # Global pooling to reduce to 256 features
        self.global_pooling = nn.AdaptiveAvgPool1d(features)

        # Final fully connected layer to produce timestamp percentage
        self.fc = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        encoder_outputs = []
        
        # Encoder (FPN Backbone)
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)  # Downsample after each block
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # FPN: Get features at different scales
        fpn_outputs = []
        for i in range(self.num_pools):
            fpn_output = self.fpn_blocks[i](encoder_outputs[i])
            fpn_outputs.append(fpn_output)
        
        # Concatenate features from different scales
        fused_features = torch.cat(fpn_outputs, dim=2)
        
        # Apply SPP (Spatial Pyramid Pooling) to fuse multi-scale information
        spp_features = [pool(fused_features) for pool in self.spp]
        
        # Flatten the pooled features from different pyramid levels
        spp_features_flat = torch.cat([feat.view(feat.size(0), -1) for feat in spp_features], dim=1)
        
        # Global pooling to 256 features
        x = self.global_pooling(spp_features_flat.unsqueeze(2))
        
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layers for regression (timestamp prediction)
        x = self.fc(x)
        
        return x

class SimpleCNNLocalizer(nn.Module):
    def __init__(self, num_layers=3, in_channels=1, mid_channels=64, kernel_size=3):
        super(SimpleCNNLocalizer, self).__init__()
        
        self.num_layers = num_layers
        
        # List of convolutional layers and max pooling layers
        self.conv_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.AvgPool1d(kernel_size=2, stride=2)
            ))
            in_channels = mid_channels  # Keep channels constant after the first layer
        
        # Global pooling to reduce to a fixed number of features (1 feature per channel)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers for regression
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.5),  # Apply dropout to reduce overfitting
            nn.Linear(mid_channels, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),  # Apply dropout to reduce overfitting
            nn.Linear(64, 1),
            nn.Sigmoid(),  # If labels are between 0 and 1
        )
        self.init_weights()
        
    def init_weights(self):
        # Iterate through the model layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # He initialization for ReLU
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Pass through the convolutional layers with max pooling
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global pooling to reduce to 1 feature per channel
        x = self.global_pool(x)
        
        # Flatten the features before passing them to the MLP
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers for final output
        x = self.fc(x)
        
        return x

if __name__ == "__main__":
    # Example usage:
    input_size = 8_000_000  # Assuming 8 million time steps in the input sequence
    model = SimpleCNNLocalizer().cuda()

    # Example data (batch_size=1, channels=1, input_size=8 million)
    data = torch.randn(1, 1, input_size).cuda()

    # Forward pass
    output = model(data)
    print(f"Predicted timestamp as percentage: {output.item()}")



# if __name__=="__main__":
#     # Example usage:
#     # Define the model
#     input_size = 8_000_000  # Assuming 8 million time steps in the input sequence
#     model = FPN1DLocalizer(input_size).cuda()

#     # Example data (batch_size=1, channels=1, input_size=8 million)
#     data = torch.randn(1, 1, input_size).cuda()

#     # Forward pass
#     output = model(data)
#     print(f"Predicted timestamp as percentage: {output.item()}")
