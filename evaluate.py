import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from autoencoder.models import Autoencoder, INPUT_BITS
from train import generate_data

def calculate_ber(y_true, y_pred_logits):
    """Calculates the Bit Error Rate (BER)."""
    y_pred_bits = (torch.sigmoid(y_pred_logits) > 0.5).float()
    return (y_true != y_pred_bits).float().mean().item()

def main():
    parser = argparse.ArgumentParser(description="Evaluate the CNN autoencoder for OFDM.")
    parser.add_argument("--model-path", type=str, default="autoencoder_cnn.pth", help="Path to the trained CNN model.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for evaluation.")
    parser.add_argument("--num-batches", type=int, default=100, help="Number of batches to test per SNR point.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = Autoencoder()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    # --- Plot Constellation for one block ---
    with torch.no_grad():
        sample_bits = generate_data(1).to(device)
        transmitted_symbols = model.transmitter(sample_bits)
        normalized_symbols = model.normalization(transmitted_symbols)
        constellation = normalized_symbols.cpu().numpy().reshape(-1, 2)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(constellation[:, 0], constellation[:, 1], alpha=0.5)
        plt.title('Learned Constellation for one 1024-bit block')
        plt.xlabel('I Component')
        plt.ylabel('Q Component')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--')
        plt.gca().add_artist(circle)
        plt.savefig("constellation_cnn.png")
        print("Constellation plot for one block saved to constellation_cnn.png")

    # --- BER vs. SNR Evaluation ---
    snr_range_db = np.arange(-5, 16, 1) # More granular SNR range
    ber_values = []

    with torch.no_grad():
        for snr_db in tqdm(snr_range_db, desc="Evaluating BER vs. SNR"):
            total_ber = 0
            for _ in range(args.num_batches):
                inputs = generate_data(args.batch_size).to(device)
                snr_tensor = torch.full((args.batch_size, 1), float(snr_db), device=device)
                
                outputs = model(inputs, snr_tensor)
                total_ber += calculate_ber(inputs, outputs)
            
            ber_values.append(total_ber / args.num_batches)

    plt.figure(figsize=(10, 6))
    plt.plot(snr_range_db, ber_values, 'bo-', label='CNN Autoencoder (BER)')
    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs. SNR for CNN-based Autoencoder')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.savefig("ber_vs_snr_cnn.png")
    print("Evaluation complete. Plot saved to ber_vs_snr_cnn.png")

if __name__ == "__main__":
    main()