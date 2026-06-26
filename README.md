# Ground2Satellite: Cross-View Image Synthesis

<p align="center">
  <img src="readme%20assets/COVER%20G2A.gif" alt="Ground-to-satellite image synthesis" width="75%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" height="28" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" height="28" />
  <img src="https://img.shields.io/badge/CUDA-GPU_Required-76B900?logo=nvidia&logoColor=white" alt="CUDA GPU Required" height="28" />
  <img src="https://img.shields.io/badge/Trained_on-Kaggle-20BEFF?logo=kaggle&logoColor=white" alt="Trained on Kaggle" height="28" />
  <img src="https://img.shields.io/badge/Tracking-WandB-FFBE00?logo=weightsandbiases&logoColor=black" alt="WandB" height="28" />
  <img src="https://img.shields.io/badge/Architecture-VQGAN%20%2B%20GPT-8A2BE2" alt="VQGAN + GPT" height="28" />
</p>

**Generate satellite images from ground-level street photos using a two-stage VQGAN + Transformer architecture.**

*Alessandro Carotenuto*

Built on top of [Taming Transformers](https://github.com/CompVis/taming-transformers) by Esser et al. The VQGAN and minGPT Transformer are taken directly from that repository.

---

## Table of Contents

- [Results](#results)
- [Training Curves](#training-curves)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Optional: offline dataset encoding](#optional-offline-dataset-encoding)
- [Requirements](#requirements)
- [Credits](#credits)

---

## Results

Each row shows a ground-level panorama (left), the model's generated satellite image (centre), and the true satellite image (right). LPIPS scores are computed against the true satellite.

<p align="center">
  <img src="readme%20assets/G2A%202.png" alt="Generation examples, forest road and mountain road" width="48%" />
  <img src="readme%20assets/G2A%203.png" alt="Generation examples, farmland and lakeside road" width="48%" />
</p>

---

## Training Curves

Tracked via WandB over 100 epochs (RoPE + ReduceLROnPlateau, codebook 1024):

<p align="center">
  <img src="readme%20assets/G2A%204.png" alt="Training curves, loss, accuracy, learning rate, perplexity" width="90%" />
</p>

---

## How it works

The model combines two components:

- **VQGAN** (pretrained on ImageNet, frozen), encodes 256×256 images into a 16×16 grid of discrete codebook indices (256 tokens) and decodes them back. Two backbone sizes are supported:
  - `BBSIZE.LARGE`, f16, codebook 16384 (`vqgan_imagenet_f16_16384`)
  - `BBSIZE.SMALL`, f16, codebook 1024 (`vqgan_imagenet_f16_1024`)
- **GPT Transformer** (minGPT), takes 256 ground-level tokens as context and autoregressively predicts 256 satellite tokens. Supports learned absolute positional embeddings or RoPE (`USE_ROPE`).

At inference time, a street photo is encoded into 256 tokens, the transformer autoregressively generates 256 satellite tokens, and the VQGAN decodes them into the output image.

---

## Project structure

```
.
├── config.py               # User configuration (edit this)
├── train_transformer.py    # Training entry point
├── inference.py            # Single-image and batch inference
├── taming_interface.py     # Model building, checkpointing, VQGAN interface
├── CVUSA_Manager.py        # Dataset loading and preprocessing utilities
├── encoding.py             # Offline dataset pre-encoding to VQGAN tokens
├── fixermodule.py          # Compatibility fixes for taming-transformers
├── setup.py                # Environment setup script
├── kaggleNotebook.ipynb    # Kaggle training/inference notebook
├── requirements.txt        # Dependencies
└── taming-transformers/    # Cloned automatically by setup.py
```

---

## Dataset

**The dataset used to train this model is not publicly available.**

This project was trained on a custom subset of **35,191 image pairs** extracted from the [CVUSA dataset](https://mvrl.cse.wustl.edu/datasets/cvusa/). CVUSA is distributed exclusively by its authors to researchers upon request and is not freely downloadable. I do not redistribute my subset, as doing so would violate the dataset's access terms.

### How to obtain the data

1. **Request access to CVUSA** from the original authors at:
   [https://mvrl.cse.wustl.edu/datasets/cvusa/request_access.html](https://mvrl.cse.wustl.edu/datasets/cvusa/request_access.html)

2. **Build your own subset** from the full dataset, following the structure described below. The code expects image pairs (ground-level panorama + polar-projected aerial image) organised in a specific folder and CSV layout.

### Expected dataset structure

```
CVUSA_subset_35191_raw/
├── train.csv           # training split index
├── val.csv             # validation split index
├── streetview/         # ground-level panoramic photos  (.jpg)
└── polarmap/           # polar-projected aerial images  (.png)
```

Each CSV has a header row followed by one pair per line:

```
satellite_path,ground_path
polarmap/0000001.png,streetview/0000001.jpg
polarmap/0000002.png,streetview/0000002.jpg
...
```

- **Column 0**, relative path to the polar satellite image (from `DATA_ROOT`)
- **Column 1**, relative path to the ground-level image (from `DATA_ROOT`)

Point `DATA_ROOT` in `config.py` to your dataset root once the folder is ready:

```python
DATA_ROOT = "/path/to/your/CVUSA_subset_raw"
```

### Adapting to a different CSV layout

If your CSV has a different column order or extra columns, change only these two lines in `CVUSA_Manager.py`, inside `CVUSADataset.__init__`:

```python
# Default (col 0 = polar, col 1 = ground):
polar_rel  = row[0].replace('\\', os.sep).strip()
ground_rel = row[1].replace('\\', os.sep).strip()
```

Everything else (path joining, transforms, dataloaders) is generic and requires no other changes.

### Dataloader API

```python
from CVUSA_Manager import CVUSADataset

train_loader, test_loader = CVUSADataset.create_dataloaders(
    data_root="/path/to/your/dataset",
    batch_size=8,
    train_csv="/path/to/your/dataset/train.csv",  # optional, defaults to data_root/train.csv
    test_csv="/path/to/your/dataset/val.csv",      # optional, defaults to data_root/val.csv
)
```

---

## Setup

**Local:**
```bash
python setup.py
```

**Kaggle:**
```python
!git clone https://github.com/Alessandro-Carotenuto/synt_satellite_vqgan_from_TT
%cd synt_satellite_vqgan_from_TT
!python setup.py
```

This installs all dependencies and clones taming-transformers. The VQGAN weights are downloaded automatically on the first run (~2 GB).

On Kaggle, upload your prepared dataset as a private dataset and point `DATA_ROOT` to the mounted path shown in the Kaggle UI.

---

## Configuration

All user settings live in `config.py`:

```python
KAGGLE_FLAG = "KAGGLE_KERNEL_RUN_TYPE" in os.environ  # auto-detected, do not change

if KAGGLE_FLAG:
    DATA_ROOT = "/kaggle/input/datasets/<your-username>/<your-dataset-name>"
else:
    DATA_ROOT = "path/to/your/CVUSA_subset_raw"  # update this

# Training
NUM_EPOCHS                      = 30
LEARNING_RATE_MODE              = LRMODE.COSINEANNEALING  # FIXED | COSINEANNEALING | COSINEANNEALING_WR | REDUCEONPLATEAU
WARM_RESTART_CYCLE              = 10                       # used only with COSINEANNEALING_WR
LEARNING_RATE                   = 5e-4
BATCH_SIZE                      = 8
DROPOUT                         = 0.25
PATIENCE_FOR_LRREDUCEONPLATEAU  = 2                        # used only with REDUCEONPLATEAU

# Token masking: pkeep linearly decays during training
TOKEN_MASKING_SCHEDULING_START  = 1.0
TOKEN_MASKING_SCHEDULING_END    = 0.8

# Backbone
BACKBONE_SIZE = BBSIZE.LARGE  # LARGE = 16384 codebook | SMALL = 1024 codebook

# Architecture
HEADS    = 8
LAYERS   = 6
USE_ROPE = False  # True: RoPE positional encoding (requires retraining from scratch)

# Experiment tracking (optional)
USE_WANDB   = False
RUN_NAME    = "default_run_name"
WANDB_GROUP = "default_group_name"
IDENTIFIER  = "more_info"

# Inference
INFERENCE_FROM = 0
INFERENCE_TO   = 5
TEMPERATURE    = 0.7
TOP_K          = 10
TOP_P          = 0.85
```

Override any setting before training without editing `config.py` directly:

```python
import config
config.NUM_EPOCHS = 100
config.BATCH_SIZE = 16
```

---

## Training

**Local:**
```bash
python train_transformer.py
```

**Kaggle (notebook):**
```python
import train_transformer
train_transformer.main()
```

### Resuming from a checkpoint

```python
import config
import train_transformer
from taming_interface import load_with_optimizer
from CVUSA_Manager import CVUSADataset

RESUME_CHECKPOINT = "CVUSAGround2Satellite_improved_epoch60_loss4.409_....pth"

model, optimizer, scheduler, checkpoint_info, device = load_with_optimizer(
    checkpoint_path=RESUME_CHECKPOINT,
    kaggle_flag=config.KAGGLE_FLAG,
    lr=config.LEARNING_RATE,
)
print(f"Resumed from epoch {checkpoint_info['epoch']}, loss {checkpoint_info['loss']:.4f}")

train_loader, test_loader = CVUSADataset.create_dataloaders(
    data_root=config.DATA_ROOT,
    batch_size=config.BATCH_SIZE,
)

train_transformer.train_model_with_evaluation(
    model, train_loader, test_loader,
    num_epochs=config.NUM_EPOCHS,
    lr=config.LEARNING_RATE,
    optimizer=optimizer,
    scheduler=scheduler,
    start_epoch=checkpoint_info['epoch'],
)
```

### Checkpoints

```
CVUSAGround2Satellite_improved_epoch{N}_loss{L}_{timestamp}.pth
CVUSAGround2Satellite_routine_epoch{N}_loss{L}_{timestamp}.pth
```

- **Improved**, saved when test loss or top-10 accuracy improves (at most once per epoch); the previous improved checkpoint is deleted.
- **Routine**, saved every 5 epochs, never deleted.

Only transformer weights are saved. The VQGAN is frozen and reconstructed from the public checkpoint at load time.

### Training features

| Feature | Detail |
|---|---|
| Mixed precision (AMP) | Automatic via `GradScaler` |
| Gradient clipping | Max norm 1.0 |
| Label smoothing | 0.1, cross-entropy (disabled at eval) |
| Selective weight decay | Applied to linear weights; biases, LayerNorm, embeddings excluded |
| Token masking | `pkeep` linearly decays from `TOKEN_MASKING_SCHEDULING_START` to `TOKEN_MASKING_SCHEDULING_END` |
| LR scheduling | Configurable via `LEARNING_RATE_MODE` |
| Data augmentation | Synchronized horizontal flip (train only); color jitter on ground image only |

### Evaluation metrics

Reported every epoch:

| Metric | Description |
|---|---|
| Train / Test loss | Cross-entropy (no label smoothing at eval) |
| Top-1 accuracy | Fraction of tokens predicted exactly |
| Top-10 accuracy | Fraction of tokens where the correct token is in the top-10 predictions |
| Perplexity | `exp(test_loss)` |
| Loss gap | `abs(test_loss - train_loss)`, widens under overfitting |

### WandB integration

Set `USE_WANDB = True` in `config.py`. On Kaggle, store your API key as a secret named `WANDB_API_KEY`. Locally, create a `.env` file:

```
WANDB_API_KEY=your_key_here
```

---

## Inference

### Load a checkpoint

```python
from taming_interface import load_saved_model, find_latest_checkpoint
import config

path = find_latest_checkpoint("CVUSAGround2Satellite_improved")
model, checkpoint_info, device = load_saved_model(path, kaggle_flag=config.KAGGLE_FLAG)
model.eval()
```

### Single image

```python
from inference import single_image_inference

generated_pil, ground_pil = single_image_inference(
    model,
    "path/to/streetview/0000001.jpg",
    real_polar_path="path/to/polarmap/0000001.png",  # optional; adds target column to display
    device=device,
    temperature=0.7,
    top_k=10,
    top_p=0.85,
)
```

### Batch test on the validation set

```python
from inference import test_inference
import config

# Runs on val.csv rows [INFERENCE_FROM, INFERENCE_TO)
test_inference(model, data_root=config.DATA_ROOT, device=device)
```

---

## Optional: offline dataset encoding

`encoding.py` pre-encodes the full dataset to VQGAN token files (`.pt` tensors of 256 `int16` indices), eliminating the encoder forward pass from every training batch. The output directory mirrors the source structure with images replaced by `.pt` files; CSVs are rewritten to match.

See the module docstring at the top of `encoding.py` for full usage details.

---

## Requirements

- Python **3.10+** (structural `match` statements are used throughout)
- CUDA-capable GPU strongly recommended
- [taming-transformers](https://github.com/CompVis/taming-transformers), installed automatically by `setup.py`

Key packages: see `requirements.txt`.

---

## Credits

- [Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers), Esser et al., 2021
- [CVUSA Dataset](https://mvrl.cse.wustl.edu/datasets/cvusa/), Zhai et al., 2017
