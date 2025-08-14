import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from autoencoder.models import Autoencoder, NUM_MESSAGES
from train import generate_data

def calculate_ser(y_true, y_pred_logits):
    """Calculates the Symbol Error Rate (SER)."""
    y_pred = torch.argmax(y_pred_logits, dim=1)
    return (y_true != y_pred).float().mean().item()

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
        # Generate all possible messages to see the full constellation
        all_messages = torch.arange(0, NUM_MESSAGES, device=device)
        constellation_points = model.transmitter(all_messages)
        constellation_points = model.normalization(constellation_points)
        constellation = constellation_points.cpu().numpy().reshape(-1, 2)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(constellation[:, 0], constellation[:, 1])
        plt.title(f'Learned Constellation Diagram ({NUM_MESSAGES} points)')
        plt.xlabel('I Component')
        plt.ylabel('Q Component')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--')
        plt.gca().add_artist(circle)
        plt.savefig("constellation.png")
        print("Constellation plot saved to constellation.png")

    # --- SER vs. SNR Evaluation ---
    snr_range_db = np.arange(-10, 21, 2)
    ser_values = []

    with torch.no_grad():
        for snr_db in tqdm(snr_range_db, desc="Evaluating SER vs. SNR"):
            total_ser = 0
            for _ in range(args.num_batches):
                inputs = generate_data(args.batch_size).to(device)
                snr_tensor = torch.full((args.batch_size, 1), float(snr_db), device=device)
                
                outputs = model(inputs, snr_tensor)
                total_ser += calculate_ser(inputs, outputs)
            
            ser_values.append(total_ser / args.num_batches)

    plt.figure(figsize=(10, 6))
    plt.plot(snr_range_db, ser_values, 'bo-', label='Autoencoder (SER)')
    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title('SER vs. SNR for Message-based Autoencoder')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.savefig("ser_vs_snr.png")
    print("Evaluation complete. Plot saved to ser_vs_snr.png")

if __name__ == "__main__":
    main()