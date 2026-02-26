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
└── CVUSA_subset/           # Dataset directory (local only, see below)
```

---

## Dataset

Use the pre-processed dataset available on Kaggle, with CSV files already fixed:

**[CVUSA Subset — CSV Fixed](https://www.kaggle.com/datasets/carotenutoalessandro/cvusa-subset-csvfixed)**

### Local setup

Download the dataset and place it in a `CVUSA_subset_csvfixed/` folder in the project root:

```
CVUSA_subset_csvfixed/
├── train-19zl_fixed.csv
├── val-19zl_fixed.csv
├── streetview/
├── bingmap/
└── polarmap/
```

### Kaggle setup

Import the dataset in your Kaggle notebook. It will be available automatically at `/kaggle/input/cvusa-subset-csvfixed` — no further steps needed.

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
    DATA_ROOT = "/kaggle/input/cvusa-subset-csvfixed"
else:
    DATA_ROOT = "CVUSA_subset_csvfixed"

NUM_EPOCHS    = 75
LEARNING_RATE = 5e-4
BATCH_SIZE    = 8
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
    "CVUSA_subset_csvfixed/streetview/0000001.jpg",
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