import torch
from torch import nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    """Decoder block with upsampling"""

    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size, dropout):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, pool_size, stride=pool_size)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle sequence length differences from pooling
        diff_h = skip.size(2) - x.size(2)
        half_h = diff_h // 2
        diff_w = skip.size(3) - x.size(3)
        half_w = diff_w // 2
        x = F.pad(
            x, (half_h, diff_h - half_h, half_w, diff_w - half_w)
        )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DownBlock(nn.Module):
    """Encoder block with downsampling"""

    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        skip = x  # Save for skip connection
        x = self.pool(x)
        return x, skip

class UNet2D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=None,
                 hidden_dim=32,
                 num_layers=3,
                 kernel_size=3,
                 dropout=0.1,
                 pool_size=2
    ):
        """
        2D U-Net with adaptive depth and sequence handling

        Args:
            in_channels: Input channels/features
            hidden_dim: Base hidden dimension (doubles with each downsampling)
            num_layers: Number of down/up sampling layers (total depth = num_layers + 1)
            kernel_size: Convolution kernel size (should be odd)
            dropout: Dropout probability
            num_classes: Number of output classes
            pool_size: Pooling/upsampling factor
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = out_channels if out_channels else in_channels
        self.hidden_dim = hidden_dim
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
            nn.Conv2d(current_channels, hidden_dim * (2 ** num_layers),
                      kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_dim * (2 ** num_layers)),
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

        self.out_conv = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)

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
        # 1x1 convolution for output
        x = self.out_conv(x)
        return x

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    in_channels = 1
    out_channels = 1
    num_layers = 3
    pool_size = 2
    channel_width = 32

    a = torch.randn((1, in_channels, 32, 32), device=device)
    model = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        pool_size=pool_size
    )
    print(model)
    model.to(device)

    with torch.no_grad():
        out = model(a)

    print(out.shape)
    print(sum(p.numel() for p in model.parameters()))

