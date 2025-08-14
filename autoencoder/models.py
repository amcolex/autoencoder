import torch
import torch.nn as nn
from .layers import Normalization, AWGN

# --- Model Configuration ---
INPUT_BITS = 1024
N_CHANNEL = 1024

class Transmitter(nn.Module):
    """
    A deep 1D CNN-based encoder with Batch Normalization.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.encoder(x)
        return x.view(x.shape[0], -1)

class Receiver(nn.Module):
    """
    A deep 1D CNN-based decoder with Batch Normalization.
    """
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(2, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 1, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 2, N_CHANNEL)
        x = self.decoder(x)
        return x.squeeze(1)

class Autoencoder(nn.Module):
    """
    The complete autoencoder model, combining the deep CNN transmitter, 
    channel, and deep CNN receiver.
    """
    def __init__(self):
        super().__init__()
        self.transmitter = Transmitter()
        self.normalization = Normalization(N_CHANNEL)
        self.channel = AWGN()
        self.receiver = Receiver()

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        x = self.transmitter(x)
        x = self.normalization(x)
        x = self.channel(x, snr_db)
        x = self.receiver(x)
        return x