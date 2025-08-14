import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from autoencoder.models import Autoencoder, INPUT_BITS

def generate_data(batch_size):
    """Generates a batch of random bits."""
    return torch.randint(0, 2, (batch_size, INPUT_BITS), dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Train the CNN autoencoder for OFDM.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (smaller for larger models).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--start-snr", type=float, default=20.0, help="Starting SNR in dB for curriculum learning.")
    parser.add_argument("--end-snr", type=float, default=0.0, help="Ending SNR in dB for curriculum learning.")
    parser.add_argument("--save-path", type=str, default="autoencoder_cnn.pth", help="Path to save the trained model.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = Autoencoder().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        current_snr = args.start_snr - (args.start_snr - args.end_snr) * (epoch / (args.epochs -1))

        with tqdm(range(1000), unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{args.epochs} (SNR: {current_snr:.2f} dB)")
            for _ in tepoch:
                inputs = generate_data(args.batch_size).to(device)
                snr_db = torch.full((args.batch_size, 1), current_snr, device=device)

                optimizer.zero_grad()
                outputs = model(inputs, snr_db)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))
        
        # Update the learning rate scheduler based on the epoch's average loss
        epoch_loss = running_loss / 1000
        scheduler.step(epoch_loss)
    
    print("Finished Training")
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()