# --- Model Configuration ---
INPUT_BITS = 1024
N_CHANNEL = 1024

# --- Training Configuration ---
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
START_SNR = 20.0
END_SNR = 0.0
MODEL_SAVE_PATH = "autoencoder_cnn.pth"

# --- Evaluation Configuration ---
EVAL_BATCH_SIZE = 1000
NUM_BATCHES = 100
CONSTELLATION_SAVE_PATH = "constellation_cnn.png"
BER_SAVE_PATH = "ber_vs_snr_cnn.png"