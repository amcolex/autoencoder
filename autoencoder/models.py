import torch
import torch.nn as nn
from .layers import Normalization, AWGN

# --- Model Configuration ---
# We now use a block-based approach with 1D CNNs.
INPUT_BITS = 1000
N_CHANNEL = 1000

class Transmitter(nn.Module):
    """
    A 1D CNN-based encoder that maps a block of bits to I/Q symbols.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # We treat the bit sequence as a single channel input
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv1d(128, 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape bit vector to (batch_size, 1, num_bits) for Conv1D
        x = x.unsqueeze(1)
        x = self.encoder(x)
        # Reshape to (batch_size, n_channel * 2)
        return x.view(x.shape[0], -1)

class Receiver(nn.Module):
    """
    A 1D CNN-based decoder to reconstruct the bits from noisy symbols.
    """
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # We use transposed convolutions to upsample back to the original size
            nn.ConvTranspose1d(2, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 1, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape I/Q vector to (batch_size, 2, n_channel) for ConvTranspose1d
        x = x.view(x.shape[0], 2, N_CHANNEL)
        x = self.decoder(x)
        # Squeeze the channel dimension to get logits
        return x.squeeze(1)

class Autoencoder(nn.Module):
    """
    The complete autoencoder model, combining the CNN transmitter, channel,
    and CNN receiver.
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