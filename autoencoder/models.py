import torch
import torch.nn as nn
from .layers import Normalization, AWGN

class Transmitter(nn.Module):
    """
    The Encoder part of the autoencoder. It learns to map the information bits
    to complex symbols for transmission.

    Args:
        input_bits (int): The number of information bits (M).
        n_channel (int): The number of complex symbols (N).
    """
    def __init__(self, input_bits: int = 100, n_channel: int = 200):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_bits, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, n_channel * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transmitter.
        """
        x = self.encoder(x)
        return x

class Receiver(nn.Module):
    """
    The Decoder part of the autoencoder. It learns to decode the received
    symbols back into the original information bits.

    Args:
        output_bits (int): The number of information bits (M).
        n_channel (int): The number of complex symbols (N).
    """
    def __init__(self, output_bits: int = 100, n_channel: int = 200):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(n_channel * 2, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, output_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the receiver.
        """
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    """
    The complete autoencoder model, combining the transmitter, channel,
    and receiver.

    Args:
        input_bits (int): The number of information bits (M).
        n_channel (int): The number of complex symbols (N).
    """
    def __init__(self, input_bits: int = 100, n_channel: int = 200):
        super().__init__()
        self.transmitter = Transmitter(input_bits, n_channel)
        self.normalization = Normalization(n_channel)
        self.channel = AWGN()
        self.receiver = Receiver(input_bits, n_channel)

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder.
        """
        x = self.transmitter(x)
        x = self.normalization(x)
        x = self.channel(x, snr_db)
        x = self.receiver(x)
        return x