import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from autoencoder.models import Autoencoder
from train import generate_data

def calculate_ber(y_true, y_pred):
    """Calculates the Bit Error Rate."""
    y_pred_bits = (torch.sigmoid(y_pred) > 0.5).float()
    return (y_true != y_pred_bits).float().mean().item()

def main():
    parser = argparse.ArgumentParser(description="Evaluate the autoencoder for OFDM.")
    parser.add_argument("--model-path", type=str, default="autoencoder.pth", help="Path to the trained model.")
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

    # --- Plot Constellation ---
    with torch.no_grad():
        # Generate a SINGLE 100-bit input vector
        sample_bits = generate_data(1, 100).to(device)
        
        # Get the constellation points from the transmitter
        constellation_points = model.transmitter(sample_bits)
        constellation_points = model.normalization(constellation_points)
        
        # Move to CPU for plotting and reshape
        constellation = constellation_points.cpu().numpy().reshape(-1, 2)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(constellation[:, 0], constellation[:, 1], alpha=0.3)
        plt.title('Learned Constellation Diagram')
        plt.xlabel('I Component')
        plt.ylabel('Q Component')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        # Draw the unit circle for reference
        circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--')
        plt.gca().add_artist(circle)
        plt.savefig("constellation.png")
        print("Constellation plot saved to constellation.png")


    # --- BER vs. SNR Evaluation ---
    snr_range_db = np.arange(-10, 21, 2)
    ber_values = []

    with torch.no_grad():
        for snr_db in tqdm(snr_range_db, desc="Evaluating BER vs. SNR"):
            total_ber = 0
            for _ in range(args.num_batches):
                inputs = generate_data(args.batch_size, 100).to(device)
                snr_tensor = torch.full((args.batch_size, 1), float(snr_db), device=device)
                
                outputs = model(inputs, snr_tensor)
                total_ber += calculate_ber(inputs, outputs)
            
            ber_values.append(total_ber / args.num_batches)

    plt.figure(figsize=(10, 6))
    plt.plot(snr_range_db, ber_values, 'bo-', label='Autoencoder')
    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs. SNR for the Autoencoder-based OFDM System')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.savefig("ber_vs_snr.png")
    print("Evaluation complete. Plot saved to ber_vs_snr.png")

if __name__ == "__main__":
    main()