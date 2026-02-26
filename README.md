# Ground2Satellite — Cross-View Image Synthesis

Generate satellite images from ground-level street photos using a VQGAN + Transformer architecture, trained on the CVUSA dataset.

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
├── CVUSA_Manager.py        # Dataset loading and preprocessing
├── fixermodule.py          # Compatibility fixes for taming-transformers
├── setup.py                # Environment setup script
├── requirements.txt        # Dependencies
└── CVUSA_subset/           # Dataset directory (see Dataset section)
```

---

## Setup

**1. Install dependencies and clone taming-transformers:**
```bash
python setup.py
```

**2. Download the CVUSA dataset** and place it in the `CVUSA_subset/` folder with this structure:
```
CVUSA_subset/
├── train-19zl.csv
├── val-19zl.csv
├── streetview/
├── bingmap/
└── polarmap/
```

**3. Preprocess the CSV files:**
```python
from CVUSA_Manager import CVUSAPreprocessor
CVUSAPreprocessor.cvusa_split_complete()
```

---

## Configuration

All user settings are in `config.py`:

```python
KAGGLE_FLAG = "KAGGLE_KERNEL_RUN_TYPE" in os.environ  # automatic, don't touch

DATA_ROOT = "CVUSA_subset"   # path to dataset

NUM_EPOCHS    = 75
LEARNING_RATE = 5e-4
BATCH_SIZE    = 8
```

---

## Training

```bash
python train_transformer.py
```

Checkpoints are saved automatically in the working directory:
- **Best model** — saved every time test loss improves, previous one deleted
- **Routine checkpoint** — saved every 5 epochs

---

## Inference

```python
from taming_interface import load_saved_model
from inference import single_image_inference

model, _, device = load_saved_model("path/to/checkpoint.pth")

single_image_inference(
    model,
    "path/to/street_photo.jpg",
    device=device,
    temperature=1.0,
    top_k=600,
    top_p=0.92,
    save_image=True
)
```

`temperature` controls randomness: lower = more deterministic, higher = more varied.

---

## Running on Kaggle

The project detects Kaggle automatically — no configuration needed. To override training parameters, add a cell at the top of your notebook **before any imports**:

```python
import config
config.NUM_EPOCHS = 100
config.BATCH_SIZE = 16
```

Then run:
```python
import train_transformer
train_transformer.main()
```

---

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended
- [taming-transformers](https://github.com/CompVis/taming-transformers) (installed automatically by `setup.py`)