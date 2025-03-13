import torch
import torch.nn as nn
import torch.nn.functional as F

ndim=6
nstep = 50

class Const_Net(nn.Module):
    def __init__(self):
        super(Const_Net, self).__init__()
        self.fc1 = nn.Linear(ndim * nstep, 10000)
        self.fc2 = nn.Linear(10000, 1000)
        self.fc3 = nn.Linear(1000, ndim * nstep)
        self.batchnorm1 = nn.BatchNorm1d(10000)
        self.batchnorm2 = nn.BatchNorm1d(1000)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = torch.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.fc3(x)
        x = x.unflatten(1, (nstep, ndim))
        return x


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


class Conv1DSeq2SeqStressPredictor(nn.Module):
    def __init__(self, input_dim=6, output_dim=6, hidden_dim=64, kernel_size=5, num_blocks=3, dropout=0.1):
        """
        Conv1D seq-to-seq model to predict stress from strain data.

        Args:
            input_dim (int): Number of input channels (strain components).
            output_dim (int): Number of output channels (stress components).
            hidden_dim (int): Number of channels in hidden layers.
            kernel_size (int): Kernel size for convolutions.
            num_blocks (int): Number of residual blocks.
            dropout (float): Dropout probability.
        """
        super().__init__()
        padding = kernel_size // 2

        # Initial convolution block to lift input into hidden_dim
        self.initial_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        # Stack of residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim, kernel_size, dropout) for _ in range(num_blocks)]
        )

        # Final convolution layer to map back to output dimensions
        self.final_conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=1, padding=0)

    def forward(self, x:torch.Tensor):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor with shape (batch_size, input_dim, time_steps)

        Returns:
            Tensor: Output tensor with shape (batch_size, output_dim, time_steps)
        """
        x = x.permute(0, 2, 1)  # Conv1D expects (batch_size, channels, time_steps)
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = self.relu(out)
        out = self.res_blocks(out)
        out = self.final_conv(out)
        out = out.permute(0, 2, 1)
        return out

class LSTMStressPredictor(nn.Module):
    def __init__(self, input_dim=6, output_dim=6, hidden_dim=64, num_layers=2, dropout=0.1, bidirectional=False):
        """ #500 epochs maybe
        LSTM model to predict stress from strain data.

        Args:
            input_dim (int): Number of input channels (strain components).
            output_dim (int): Number of output channels (stress components).
            hidden_dim (int): Number of hidden units in LSTM.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            hidden_dim *= 2
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor with shape (batch_size, time_steps, input_dim)

        Returns:
            Tensor: Output tensor with shape (batch_size, time_steps, output_dim)
        """
        out, _ = self.lstm(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out



class LSTMStressPredictor_ln(nn.Module):
    def __init__(self, input_dim=6, output_dim=6, hidden_dim=64, num_layers=2, dropout=0.1, bidirectional=False):
        """ #500 epochs maybe
        LSTM model to predict stress from strain data.

        Args:
            input_dim (int): Number of input channels (strain components).
            output_dim (int): Number of output channels (stress components).
            hidden_dim (int): Number of hidden units in LSTM.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        if bidirectional:
            hidden_dim *= 2

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor with shape (batch_size, time_steps, input_dim)

        Returns:
            Tensor: Output tensor with shape (batch_size, time_steps, output_dim)
        """
        out, _ = self.lstm(x)
        out = self.ln(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out