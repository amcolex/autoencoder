import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from autoencoder.models import Autoencoder, NUM_MESSAGES

def generate_data(batch_size):
    """Generates a batch of random message indices."""
    return torch.randint(0, NUM_MESSAGES, (batch_size,), dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Train the autoencoder for OFDM.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--start-snr", type=float, default=20.0, help="Starting SNR in dB for curriculum learning.")
    parser.add_argument("--end-snr", type=float, default=0.0, help="Ending SNR in dB for curriculum learning.")
    parser.add_argument("--save-path", type=str, default="autoencoder.pth", help="Path to save the trained model.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = Autoencoder().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
    
    print("Finished Training")
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()