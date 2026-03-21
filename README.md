# Ground2Satellite — Cross-View Image Synthesis

Generate satellite images from ground-level street photos using a VQGAN + Transformer architecture, trained on the CVUSA dataset.

This project is built on top of [Taming Transformers](https://github.com/CompVis/taming-transformers) by Esser et al. The VQGAN and the Net2Net Transformer architecture are taken directly from that repository.

---

## How it works

The model combines two components:

- **VQGAN** (pretrained on ImageNet) — encodes images into discrete token sequences and decodes them back into images
- **GPT Transformer** (minGPT) — learns to predict satellite token sequences conditioned on ground-level tokens

The VQGAN is frozen during training. Only the transformer is trained.

At inference time, a street photo is encoded into 256 tokens, the transformer autoregressively generates 256 satellite tokens, and the VQGAN decodes them into the final image.

---

## Project structure

```
.
├── config.py               # User configuration (edit this)
├── train_transformer.py    # Training entry point
├── inference.py            # Inference functions
├── taming_interface.py     # Model building, checkpointing, VQGAN interface
├── CVUSA_Manager.py        # Dataset loading
├── fixermodule.py          # Compatibility fixes for taming-transformers
├── setup.py                # Environment setup script
├── requirements.txt        # Dependencies
└── CVUSA_subset_35191_raw/ # Dataset directory (local only, see below)
```

---

## Dataset

This project uses a subset of the CVUSA dataset containing **35,191 image pairs** (ground-level + polar satellite).

The dataset is available on Kaggle:

**[CVUSA Ground and Polar Subset — 35191](https://www.kaggle.com/datasets/carotenutoalessandro/cvusa-groundandpolar-subset-35191)**

### Local setup

Download the dataset and place it in a `CVUSA_subset_35191_raw/` folder in the project root:

```
CVUSA_subset_35191_raw/
├── train.csv
├── val.csv
├── streetview/
└── polarmap/
```

### Kaggle setup

Import the dataset in your Kaggle notebook. It will be available automatically at `/kaggle/input/datasets/carotenutoalessandro/cvusa-groundandpolar-subset-35191` — no further steps needed.

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

This installs all dependencies and clones taming-transformers. Optionally edit `config.py` before training (see Configuration).

---

## Configuration

All user settings are in `config.py`:

```python
KAGGLE_FLAG = "KAGGLE_KERNEL_RUN_TYPE" in os.environ  # automatic, don't touch

if KAGGLE_FLAG:
    DATA_ROOT = "/kaggle/input/datasets/carotenutoalessandro/cvusa-groundandpolar-subset-35191"
else:
    DATA_ROOT = "CVUSA_subset_35191_raw"

# Training
NUM_EPOCHS    = 30
LEARNING_RATE_MODE = LRMODE.FIXED   # FIXED, COSINEANNEALING, COSINEANNEALING_WR
LEARNING_RATE = 5e-4
BATCH_SIZE    = 8
DROPOUT       = 0.25

# Token masking schedule (linearly decays from START to END over training)
TOKEN_MASKING_SCHEDULING_START = 1.0
TOKEN_MASKING_SCHEDULING_END   = 0.7

# Architecture
HEADS  = 8
LAYERS = 6
```

On Kaggle, override parameters directly before running:
```python
import config
config.NUM_EPOCHS = 100   # only needed if you want to change the defaults
config.BATCH_SIZE = 16
```

---

## Training

**Local:**
```bash
python train_transformer.py
```

**Kaggle:**
```python
import train_transformer
train_transformer.main()
```

Checkpoints are saved automatically in the working directory:
- **Best model** — saved every time test loss improves, previous one deleted
- **Routine checkpoint** — saved every 5 epochs

### Training features

- **Mixed precision** (AMP) — automatic via `GradScaler`, speeds up training on CUDA
- **`torch.compile()`** — enabled by default, gives ~10–20% additional speedup
- **Gradient clipping** — max norm 1.0
- **Label smoothing** — 0.1, applied to cross-entropy loss
- **Selective weight decay** — applied only to linear weights; biases, LayerNorm and embeddings are excluded
- **Token masking** — linearly scheduled from `TOKEN_MASKING_SCHEDULING_START` to `TOKEN_MASKING_SCHEDULING_END` over all epochs
- **LR scheduling** — configurable via `LEARNING_RATE_MODE`: `FIXED`, `COSINEANNEALING`, or `COSINEANNEALING_WR`

---

## Inference

### Load a saved model
```python
from taming_interface import load_saved_model

model, _, device = load_saved_model("checkpoints/my_checkpoint.pth")
```

### Single image
```python
from inference import single_image_inference

single_image_inference(
    model,
    "CVUSA_subset_35191_raw/streetview/0000001.jpg",
    device=device,
    temperature=1.0,   # lower = more deterministic, higher = more varied
    top_k=600,
    top_p=0.92,
    save_image=True
)
```

### Quick test on val set
Runs inference on the first 5 images of the validation set:
```python
from inference import test_inference
import config

test_inference(model, data_root=config.DATA_ROOT, device=device)
```

---

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended
- [taming-transformers](https://github.com/CompVis/taming-transformers) (installed automatically by `setup.py`)

---

## Credits

- [Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers) — Esser et al., 2021
- [CVUSA Dataset](https://mvrl.cse.wustl.edu/datasets/cvusa/) — Zhai et al., 2017