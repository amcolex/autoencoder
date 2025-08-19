import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from autoencoder.models import Autoencoder, INPUT_BITS
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import config
import random

def generate_data(batch_size):
    """Generates a batch of random bits."""
    return torch.randint(0, 2, (batch_size, INPUT_BITS), dtype=torch.float32)

def calculate_ber(y_true, y_pred_logits):
    """Calculates the Bit Error Rate (BER)."""
    y_pred_bits = (torch.sigmoid(y_pred_logits) > 0.5).float()
    return (y_true != y_pred_bits).float().mean().item()

def plot_constellation(model, device, sample_bits, path):
    """Plots the constellation diagram for a given model and data."""
    model.eval()
    with torch.no_grad():
        transmitted_symbols = model.transmitter(sample_bits)
        normalized_symbols = model.normalization(transmitted_symbols)
        constellation = normalized_symbols.cpu().numpy().reshape(-1, 2)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(constellation[:, 0], constellation[:, 1], alpha=0.5)
        plt.title('Learned Constellation')
        plt.xlabel('I Component')
        plt.ylabel('Q Component')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--')
        plt.gca().add_artist(circle)
        plt.savefig(path)
        plt.close()

def plot_ber_vs_snr(model, device, path, num_batches, batch_size):
    """Plots the BER vs. SNR curve."""
    model.eval()
    snr_range_db = np.arange(-5, 16, 1)
    ber_values = []

    with torch.no_grad():
        for snr_db in snr_range_db:
            total_ber = 0
            for _ in range(num_batches):
                inputs = generate_data(batch_size).to(device)
                snr_tensor = torch.full((batch_size, 1), float(snr_db), device=device)
                outputs = model(inputs, snr_tensor)
                total_ber += calculate_ber(inputs, outputs)
            ber_values.append(total_ber / num_batches)

    plt.figure(figsize=(10, 6))
    plt.plot(snr_range_db, ber_values, 'bo-', label='CNN Autoencoder (BER)')
    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER vs. SNR')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.ylim(1e-6, 1)
    plt.savefig(path)
    plt.close()

def main():
    # --- Create run directory ---
    run_dir = os.path.join("runs", config.RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved in: {run_dir}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = Autoencoder().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    # Generate a fixed sample for consistent constellation plots
    constellation_sample = generate_data(1).to(device)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        with tqdm(range(1000), unit="batch") as tepoch: # Using 1000 batches per epoch
            tepoch.set_description(f"Epoch {epoch+1}/{config.EPOCHS}")
            for _ in tepoch:
                # Sample a random SNR for each batch from the specified range
                current_snr = random.uniform(config.SNR_RANGE_DB[0], config.SNR_RANGE_DB[1])
                
                inputs = generate_data(config.BATCH_SIZE).to(device)
                snr_db = torch.full((config.BATCH_SIZE, 1), current_snr, device=device)

                optimizer.zero_grad()
                outputs = model(inputs, snr_db)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))
        
        epoch_loss = running_loss / 1000
        scheduler.step(epoch_loss)
        
        # --- Save checkpoint and plots ---
        checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        constellation_path = os.path.join(run_dir, f"constellation_epoch_{epoch+1}.png")
        plot_constellation(model, device, constellation_sample, constellation_path)

        ber_plot_path = os.path.join(run_dir, f"ber_vs_snr_epoch_{epoch+1}.png")
        plot_ber_vs_snr(model, device, ber_plot_path, num_batches=config.EVAL_NUM_BATCHES, batch_size=config.EVAL_BATCH_SIZE)

        print(f"Epoch {epoch+1} finished. Checkpoint and plots saved.")

    print("Finished Training")

if __name__ == "__main__":
    main()