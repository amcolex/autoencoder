import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from autoencoder.models import Autoencoder
from autoencoder import config
from train import generate_data

def calculate_ber(y_true, y_pred_logits):
    """Calculates the Bit Error Rate (BER)."""
    y_pred_bits = (torch.sigmoid(y_pred_logits) > 0.5).float()
    return (y_true != y_pred_bits).float().mean().item()

def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = Autoencoder()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
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
        plt.savefig(config.CONSTELLATION_SAVE_PATH)
        print(f"Constellation plot for one block saved to {config.CONSTELLATION_SAVE_PATH}")

    # --- BER vs. SNR Evaluation ---
    snr_range_db = np.arange(-5, 16, 1) # More granular SNR range
    ber_values = []

    with torch.no_grad():
        for snr_db in tqdm(snr_range_db, desc="Evaluating BER vs. SNR"):
            total_ber = 0
            for _ in range(config.NUM_BATCHES):
                inputs = generate_data(config.EVAL_BATCH_SIZE).to(device)
                snr_tensor = torch.full((config.EVAL_BATCH_SIZE, 1), float(snr_db), device=device)
                
                outputs = model(inputs, snr_tensor)
                total_ber += calculate_ber(inputs, outputs)
            
            ber_values.append(total_ber / config.NUM_BATCHES)

    plt.figure(figsize=(10, 6))
    plt.plot(snr_range_db, ber_values, 'bo-', label='CNN Autoencoder (BER)')
    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs. SNR for CNN-based Autoencoder')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.savefig(config.BER_SAVE_PATH)
    print(f"Evaluation complete. Plot saved to {config.BER_SAVE_PATH}")

if __name__ == "__main__":
    main()