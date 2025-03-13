from math import floor
import torch
from torch import nn
import torch.nn.functional as F


class UNet1DClassifier(nn.Module):
    def __init__(self,
                 in_channels=1,
                 hidden_dim=32,
                 num_layers=3,
                 kernel_size=3,
                 dropout=0.2,
                 num_classes=1,
                 pool_size=2
    ):
        """
        1D U-Net inspired classifier with adaptive depth and sequence handling

        Args:
            in_channels: Input channels/features per time step
            hidden_dim: Base hidden dimension (doubles with each downsampling)
            num_layers: Number of down/up sampling layers (total depth = num_layers + 1)
            kernel_size: Convolution kernel size (should be odd)
            dropout: Dropout probability
            num_classes: Number of output classes
            pool_size: Pooling/upsampling factor
        """
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        padding = kernel_size // 2

        # Encoder path
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_layers):
            self.encoders.append(
                DownBlock(current_channels, hidden_dim * (2 ** i),
                          kernel_size, padding, pool_size, dropout)
            )
            current_channels = hidden_dim * (2 ** i)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(current_channels, hidden_dim * (2 ** num_layers),
                      kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim * (2 ** num_layers)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Decoder path
        self.decoders = nn.ModuleList()
        for i in reversed(range(num_layers)):
            current_channels = hidden_dim * (2 ** i)
            self.decoders.append(
                UpBlock(2*current_channels, current_channels,
                kernel_size, padding, pool_size, dropout)
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, channels, seq_len)

        skips = []

        # Encoder path
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Classification
        return self.classifier(x)

class DownBlock(nn.Module):
    """Encoder block with downsampling"""

    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        skip = x  # Save for skip connection
        x = self.pool(x)
        return x, skip

class UpBlock(nn.Module):
    """Decoder block with upsampling"""

    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size, dropout):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, out_channels, pool_size, stride=pool_size)
        self.conv = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle sequence length differences from pooling
        diff = skip.size(2) - x.size(2)
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.1):
        """
        A residual block with two Conv1D layers, BatchNorm and ReLU activations.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolutional kernel size (should be odd).
            dropout (float): Dropout probability.
        """
        super().__init__()
        padding = kernel_size // 2  # To preserve sequence length

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # If dimensions differ, adjust the residual connection
        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = None

    def forward(self, x):
        identity = x if self.res_conv is None else self.res_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResidualClassifier(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=64, kernel_size=5, num_blocks=3, dropout=0.1):
        """
        1D CNN classifier with residual blocks for sequence classification.

        Args:
            input_dim (int): Number of input channels/features per time step
            output_dim (int): Number of output classes
            hidden_dim (int): Number of channels in hidden layers
            kernel_size (int): Kernel size for convolutions (should be odd)
            num_blocks (int): Number of residual blocks
            dropout (float): Dropout probability
        """
        super().__init__()
        padding = kernel_size // 2

        # Initial convolution to expand to hidden dimension
        self.initial_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        # Stack of residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim, kernel_size, dropout) for _ in range(num_blocks)]
        )

        # Adaptive pooling to handle variable sequence lengths
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Final classification layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for classification.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim, seq_len)

        Returns:
            Tensor: Output logits of shape (batch_size, output_dim)
        """
        # (batch_size, seq_len)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # (batch_size, 1, seq_len)

        # Initial processing
        out = self.initial_conv(x) #bsz, hidden_dim, seq_len
        out = self.initial_bn(out)
        out = self.relu(out)

        # Residual blocks
        out = self.res_blocks(out) # bsz, hidden_dim, seq_len

        # Pooling and final classification
        out = self.pool(out)    # (batch_size, hidden_dim, 1)
        out = out.squeeze(-1)   # (batch_size, hidden_dim)
        out = self.fc(out)      # (batch_size, output_dim)
        out = self.sigmoid(out) # (batch_size, output_dim)
        return out
