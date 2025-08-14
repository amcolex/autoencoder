# End-to-End Autoencoder for OFDM

This project is a proof-of-concept for an end-to-end neural network-based autoencoder for an OFDM wireless transmission system. The autoencoder learns to perform the entire transmission and reception process, including modulation and coding, to maximize the Bit Error Rate (BER) vs. Signal-to-Noise Ratio (SNR) performance over an AWGN channel.

## Architecture

The system consists of three main components:
- **Transmitter (Encoder):** A neural network that maps 100 information bits to 200 complex OFDM symbols.
- **AWGN Channel:** A simulated channel that adds Additive White Gaussian Noise to the transmitted symbols.
- **Receiver (Decoder):** A neural network that reconstructs the original information bits from the noisy symbols.

The entire system is trained end-to-end to minimize the binary cross-entropy between the original and reconstructed bits.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- `uv` package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/autoencoder.git
   cd autoencoder
   ```

2. Install the dependencies using `uv`:
   ```bash
   uv pip install -e .
   ```

### Training

To train the autoencoder, run the `train.py` script:
```bash
python train.py --epochs 100 --snr 10
```
This will train the model for 100 epochs at an SNR of 10 dB and save the trained model to `autoencoder.pth`.

### Evaluation

To evaluate the trained model and generate a BER vs. SNR plot, run the `evaluate.py` script:
```bash
python evaluate.py --model-path autoencoder.pth
```
This will run the evaluation and save the plot to `ber_vs_snr.png`.

## Results

The expected output of the evaluation is a plot showing the Bit Error Rate as a function of the Signal-to-Noise Ratio. This plot demonstrates how well the autoencoder has learned to communicate reliably over a noisy channel.