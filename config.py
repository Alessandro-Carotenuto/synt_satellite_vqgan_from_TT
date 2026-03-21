import os
from enum import Enum

class LRMODE(Enum):
    FIXED=0
    COSINEANNEALING=1
    COSINEANNEALING_WR=2

# ========================
# USER CONFIGURATION
# ========================

# Automatically True if running on Kaggle, False otherwise
KAGGLE_FLAG = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
OLD_SUBSET=False

if KAGGLE_FLAG:
    DATA_ROOT = "/kaggle/input/datasets/carotenutoalessandro/cvusa-groundandpolar-subset-35191"
else:
    DATA_ROOT = "C:/Users/alex1/Documents/DATASETS/CVUSA_subset_35191_raw"

# Training
NUM_EPOCHS = 30
LEARNING_RATE_MODE=LRMODE.FIXED #FIXED, COSINEANNEALING, COSINEANNEALING_WR
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
DROPOUT=0.25 #25%
TOKEN_MASKING_SCHEDULING_START=1.0
TOKEN_MASKING_SCHEDULING_END=0.7

#ARCHITECTURE OPTIONS
HEADS=8     #8
LAYERS=6   #12
