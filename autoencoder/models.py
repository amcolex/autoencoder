import torch
import torch.nn as nn
from .layers import Normalization, AWGN

# --- Model Configuration ---
# We define a message vocabulary based on k bits.
# Total number of unique messages will be 2^k.
K_BITS = 7
NUM_MESSAGES = 2**K_BITS
N_CHANNEL = 200 # Number of I/Q symbols

class Transmitter(nn.Module):
    """
    The Encoder part of the autoencoder. It learns to map a message index
    to a sequence of complex symbols for transmission.
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(NUM_MESSAGES, 128)
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, N_CHANNEL * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transmitter.
        x is expected to be a tensor of message indices.
        """
        x = self.embedding(x)
        x = self.encoder(x)
        return x

class Receiver(nn.Module):
    """
    The Decoder part of the autoencoder. It learns to decode the received
    symbols back into the original message index.
    """
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(N_CHANNEL * 2, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, NUM_MESSAGES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the receiver.
        Returns logits for each possible message.
        """
        x = self.decoder(x)
        return x

class Autoencoder(nn.Module):
    """
    The complete autoencoder model, combining the transmitter, channel,
    and receiver.
    """
    def __init__(self):
        super().__init__()
        self.transmitter = Transmitter()
        self.normalization = Normalization(N_CHANNEL)
        self.channel = AWGN()
        self.receiver = Receiver()

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder.
        """
        x = self.transmitter(x)
        x = self.normalization(x)
        x = self.channel(x, snr_db)
        x = self.receiver(x)
        return x