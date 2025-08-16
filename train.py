import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from autoencoder.models import Autoencoder
from autoencoder import config

def generate_data(batch_size):
    """Generates a batch of random bits."""
    return torch.randint(0, 2, (batch_size, config.INPUT_BITS), dtype=torch.float32)

def main():

    if torch.cuda.is_available():
        # Performance optimizations for NVIDIA GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = torch.compile(Autoencoder().to(device))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        current_snr = config.START_SNR - (config.START_SNR - config.END_SNR) * (epoch / (config.EPOCHS -1))

        with tqdm(range(1000), unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{config.EPOCHS} (SNR: {current_snr:.2f} dB)")
            for _ in tepoch:
                inputs = generate_data(config.BATCH_SIZE).to(device)
                snr_db = torch.full((config.BATCH_SIZE, 1), current_snr, device=device)

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
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()