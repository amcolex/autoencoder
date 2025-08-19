# Training configuration
RUN_NAME = "mixed_snr_test_bler2"
EPOCHS = 100 # Increased epochs for better convergence
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SNR_RANGE_DB = [0, 20] # Train on SNRs from 0dB to 20dB

# Evaluation configuration
EVAL_NUM_BATCHES = 10
EVAL_BATCH_SIZE = 100