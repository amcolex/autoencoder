import torch
import torch.nn as nn

class Normalization(nn.Module):
    """
    A layer to normalize the transmitted symbols to have a maximum power of 1.
    This enforces the constraint that I^2 + Q^2 <= 1.

    Args:
        n_channel (int): The number of complex symbols (N).
    """

    def __init__(self, n_channel: int = 200):
        super().__init__()
        self.n_channel = n_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the normalization layer.

        Args:
            x: A tensor of shape (batch_size, n_channel * 2) representing the
               I/Q symbols.

        Returns:
            A tensor of shape (batch_size, n_channel * 2) with normalized
            power.
        """
        # Reshape to (batch_size, n_channel, 2) to handle I and Q pairs
        x_reshaped = x.view(-1, self.n_channel, 2)
        
        # Calculate the power of each symbol
        power = torch.sum(x_reshaped ** 2, dim=2, keepdim=True)
        
        # Normalize symbols where power > 1
        mask = (power > 1.0).float()
        x_normalized = x_reshaped / torch.sqrt(power)
        
        # Apply normalization only where needed
        output_reshaped = (1 - mask) * x_reshaped + mask * x_normalized
        
        # Reshape back to (batch_size, n_channel * 2)
        return output_reshaped.view(-1, self.n_channel * 2)

class AWGN(nn.Module):
    """
    A layer to add Additive White Gaussian Noise to the transmitted symbols.
    The noise level is controlled by the Signal-to-Noise Ratio (SNR).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AWGN channel.

        Args:
            x: A tensor of shape (batch_size, n_channel * 2) representing the
               I/Q symbols.
            snr_db: A tensor of shape (batch_size, 1) representing the SNR
                    in decibels.

        Returns:
            A tensor of shape (batch_size, n_channel * 2) representing the
            noisy symbols.
        """
        # Calculate the signal power
        signal_power = torch.mean(x ** 2, dim=1, keepdim=True)
        
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10.0)
        
        # Calculate the noise power
        noise_power = signal_power / snr_linear
        
        # Generate Gaussian noise
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        
        return x + noise