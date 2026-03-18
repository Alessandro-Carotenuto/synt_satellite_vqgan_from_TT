import os

# ========================
# USER CONFIGURATION
# ========================

# Automatically True if running on Kaggle, False otherwise
KAGGLE_FLAG = "KAGGLE_KERNEL_RUN_TYPE" in os.environ


if KAGGLE_FLAG:
    DATA_ROOT = "/kaggle/input/datasets/carotenutoalessandro/cvusa-subset-csvfixed"
else:
    DATA_ROOT = "C:/Users/alex1/Documents/DATASETS/CVUSA_subset_35191_raw"

# Training
NUM_EPOCHS = 30
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
DROPOUT=0.25 #25%
TOKEN_MASKING_SCHEDULING_START=1.0
TOKEN_MASKING_SCHEDULING_END=0.7

#ARCHITECTURE OPTIONS
HEADS=8     #8
LAYERS=6   #12
