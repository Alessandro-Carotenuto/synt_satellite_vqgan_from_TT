import os
from enum import Enum


class LRMODE(Enum):
    FIXED=0
    COSINEANNEALING=1
    COSINEANNEALING_WR=2
    REDUCEONPLATEAU=3

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
LEARNING_RATE_MODE=LRMODE.COSINEANNEALING #FIXED, COSINEANNEALING, COSINEANNEALING_WR, ReduceLROnPlateau
WARM_RESTART_CYCLE=10 
LAST_EPOCH = -1
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
DROPOUT=0.25 #25%
TOKEN_MASKING_SCHEDULING_START=1.0
TOKEN_MASKING_SCHEDULING_END=0.8
PATIENCE_FOR_LRREDUCEONPLATEAU=2

#LOGGING
USE_WANDB = False       # Set to True to enable experiment tracking (requires wandb account)
RUN_NAME = "default_run_name"
WANDB_GROUP = 'default_group_name'

#ARCHITECTURE OPTIONS
HEADS=8     #8
LAYERS=6   #12
