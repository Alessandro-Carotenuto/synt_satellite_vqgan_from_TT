import os

# ========================
# USER CONFIGURATION
# ========================

# Automatically True if running on Kaggle, False otherwise
KAGGLE_FLAG = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if KAGGLE_FLAG:
    DATA_ROOT = "/kaggle/input/datasets/carotenutoalessandro/cvusa-subset-csvfixed"
else:
    DATA_ROOT = "CVUSA_subset_csvfixed"

# Training
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
BATCH_SIZE = 8