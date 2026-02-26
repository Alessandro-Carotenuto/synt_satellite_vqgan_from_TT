import os

KAGGLE_FLAG = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
print(KAGGLE_FLAG)