 # Ground2Satellite — Cross-View Image Synthesis

  Generate satellite images from ground-level street photos using a VQGAN + Transformer architecture, trained on the
  CVUSA dataset.

  Built on top of [Taming Transformers](https://github.com/CompVis/taming-transformers) by Esser et al. The VQGAN and
  Net2Net Transformer architecture are taken directly from that repository.

  ---

  ## How it works

  The model combines two components:

  - **VQGAN** (pretrained on ImageNet, f16, codebook size 16384) — encodes 256×256 images into a 16×16 grid of discrete
  codebook indices (256 tokens) and decodes them back. Frozen during training.
  - **GPT Transformer** (minGPT) — takes 256 ground-level tokens as context and autoregressively predicts 256 satellite
  tokens.

  At inference time, a street photo is encoded into 256 tokens, the transformer autoregressively generates 256 satellite
   tokens, and the VQGAN decodes them into the output image.

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
  ├── requirements.txt        # Dependencies
  └── taming-transformers/    # Cloned automatically by setup.py
  ```

  ---

  ## Dataset

  This project trains on a **custom subset of the CVUSA dataset** containing **35,191 image pairs**, created by me. The original [CVUSA dataset](https://mvrl.cse.wustl.edu/datasets/cvusa/) is not publicly downloadable. To use it, you must request access directly from the authors, here: https://mvrl.cse.wustl.edu/datasets/cvusa/request_access.html

  ### Subset structure

  The subset used here is organised as follows:

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

  - **Column 0** — relative path to the polar satellite image (from `DATA_ROOT`)
  - **Column 1** — relative path to the ground-level image (from `DATA_ROOT`)

  Point `DATA_ROOT` in `config.py` to the dataset if you create your subset of the dataset as described above:

  ```python
  DATA_ROOT = "/path/to/CVUSA_subset_35191_raw"
  ```

  ### Using the full original CVUSA dataset

  If you have access to the original CVUSA dataset, you can use it by preparing two CSV index files (`train.csv`,
  `val.csv`) that follow the column format above, then pointing `DATA_ROOT` to your dataset root.

  In my case a polar mapping is used, so the polar image is the first column and the ground image is the second. If your CSVs have a different structure, you can adjust the column indices in `CVUSA_Manager.py` as described below.

  The only code you need to change is in `CVUSA_Manager.py`, inside `CVUSADataset.__init__`:

  ```python
  # Current (col 0 = polar, col 1 = ground):
  polar_rel  = row[0].replace('\\', os.sep).strip()
  ground_rel = row[1].replace('\\', os.sep).strip()
  ```

  If your CSV uses a different column order or more columns, adjust the indices here. Everything else (path joining,
  existence check, transforms) is generic and requires no changes.

  The dataloader entry point is:

  ```python
  from CVUSA_Manager import CVUSADataset

  train_loader, test_loader = CVUSADataset.create_dataloaders(
      data_root="/path/to/your/dataset",
      batch_size=8,
      train_csv="/path/to/your/dataset/train.csv",  # optional, defaults to data_root/train.csv
      test_csv="/path/to/your/dataset/val.csv",      # optional, defaults to data_root/val.csv
  )
  ```
  I trained this model on Kaggle and the repository is built to handle this case easily, buy you need to make your own subset of the dataset as described above and upload it to Kaggle as a private dataset. Then point `DATA_ROOT` to the mounted dataset path, which you can find on the Kaggle UI after uploading.

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

  This installs all dependencies and clones taming-transformers. 
  ---

  ## Configuration

  All user settings live in `config.py`:

  ```python
  KAGGLE_FLAG = "KAGGLE_KERNEL_RUN_TYPE" in os.environ  # auto-detected, do not change

  if KAGGLE_FLAG:
      DATA_ROOT = "/kaggle/input/datasets/carotenutoalessandro/cvusa-groundandpolar-subset-35191"
  else:
      DATA_ROOT = "path/to/your/CVUSA_subset_35191_raw"  # update this

  # Training
  NUM_EPOCHS         = 30
  LEARNING_RATE_MODE = LRMODE.COSINEANNEALING  # FIXED | COSINEANNEALING | COSINEANNEALING_WR | REDUCEONPLATEAU
  WARM_RESTART_CYCLE = 10                       # used only with COSINEANNEALING_WR
  LEARNING_RATE      = 5e-4
  BATCH_SIZE         = 8
  DROPOUT            = 0.25

  # Token masking: probability of keeping each token, linearly decays during training
  TOKEN_MASKING_SCHEDULING_START = 1.0
  TOKEN_MASKING_SCHEDULING_END   = 0.8

  # Architecture
  HEADS  = 12
  LAYERS = 6

  # Experiment tracking (optional)
  USE_WANDB   = False
  RUN_NAME    = "default_run_name"
  WANDB_GROUP = "default_group_name"

  # Inference
  INFERENCE_FROM = 0
  INFERENCE_TO   = 4
  TEMPERATURE    = 1.0
  TOP_K          = 600
  TOP_P          = 0.92
  ```
Edit `config.py` before training or override settings in a specific cell before running the training entry point.

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

  **Kaggle:**
  ```python
  import train_transformer
  train_transformer.main()
  ```

  The VQGAN weights are downloaded automatically on first run (~2 GB).

  ### Checkpoints

  Checkpoints are saved in the working directory with names like:

  ```
  CVUSAGround2Satellite_improved_epoch{N}_loss{L}_{timestamp}.pth
  CVUSAGround2Satellite_routine_epoch{N}_loss{L}_{timestamp}.pth
  ```

  - **Improved** — saved when test loss improves *or* top-10 accuracy improves (at most once per epoch); the previous
  improved checkpoint is deleted.
  - **Routine** — saved every 5 epochs, never deleted.

  Only transformer weights are saved. The VQGAN is frozen and reconstructed from the public checkpoint at load time.

  ### Training features

  | Feature | Detail |
  |---|---|
  | Mixed precision (AMP) | Automatic via `GradScaler` |
  | Gradient clipping | Max norm 1.0 |
  | Label smoothing | 0.1, cross-entropy (disabled at eval) |
  | Selective weight decay | Applied to linear weights; biases, LayerNorm, embeddings excluded |
  | Token masking | `pkeep` linearly decays from `TOKEN_MASKING_SCHEDULING_START` to `TOKEN_MASKING_SCHEDULING_END` |
  | LR scheduling | Configurable; see `LEARNING_RATE_MODE` |
  | Data augmentation | Synchronized horizontal flip (train only); color jitter on ground image only |

  ### Evaluation metrics

  Reported every epoch:

  | Metric | Description |
  |---|---|
  | Train / Test loss | Cross-entropy (no label smoothing at eval) |
  | Top-1 accuracy | Fraction of tokens predicted exactly |
  | Top-10 accuracy | Fraction of tokens where the correct token is in the top-10 predictions |
  | Perplexity | `exp(test_loss)` |
  | Loss gap | `abs(test_loss − train_loss)` — widens under overfitting |

  ### WandB integration

  Set `USE_WANDB = True` in `config.py`. On Kaggle, store your key as a secret named `WANDB_API_KEY`. Locally, create a
  `.env` file:

  ```
  WANDB_API_KEY=your_key_here
  ```

  ---

  ## Inference

  ### Load a checkpoint

  ```python
  from taming_interface import load_saved_model

  model, checkpoint, device = load_saved_model("CVUSAGround2Satellite_improved_epoch10_loss3.141_....pth")
  ```

  To find the most recent checkpoint matching a prefix automatically:

  ```python
  from taming_interface import find_latest_checkpoint

  path = find_latest_checkpoint("CVUSAGround2Satellite_improved")
  model, checkpoint, device = load_saved_model(path)
  ```

  ### Single image

  ```python
  from inference import single_image_inference

  generated_pil, ground_pil = single_image_inference(
      model,
      "path/to/streetview/0000001.jpg",
      real_polar_path="path/to/polarmap/0000001.png",  # optional; adds a target column to the display
      device=device,
      temperature=1.0,   # lower = more deterministic, higher = more varied
      top_k=600,
      top_p=0.92,
  )
  ```

  Displays a matplotlib figure and returns `(generated_pil, ground_pil)`.

  ### Batch test on the validation set

  ```python
  from inference import test_inference
  import config

  # Runs on val.csv rows [INFERENCE_FROM, INFERENCE_TO) — default: rows 0–3
  test_inference(model, data_root=config.DATA_ROOT, device=device)
  ```

  ---

  ## Resuming training

  ```python
  from taming_interface import load_with_optimizer, find_latest_checkpoint
  from train_transformer import train_model_with_evaluation
  from CVUSA_Manager import CVUSADataset
  import config

  path = find_latest_checkpoint("CVUSAGround2Satellite_improved")
  model, optimizer, scheduler, checkpoint, device = load_with_optimizer(path)

  train_loader, test_loader = CVUSADataset.create_dataloaders(
      data_root=config.DATA_ROOT,
      batch_size=config.BATCH_SIZE,
  )

  train_model_with_evaluation(
      model, train_loader, test_loader,
      num_epochs=config.NUM_EPOCHS,
      lr=config.LEARNING_RATE,
      optimizer=optimizer,
      scheduler=scheduler,
      start_epoch=checkpoint['epoch'],
  )
  ```

  ---

  ## Optional: offline dataset encoding

  `encoding.py` pre-encodes the full dataset to VQGAN token files (`.pt` tensors of 256 `int16` indices), eliminating
  the encoder forward pass from training batches. The output directory mirrors the source structure with images replaced
   by `.pt` files; CSVs are rewritten to match.

  See the module docstring at the top of `encoding.py` for full usage details.

  ---

  ## Requirements

  - Python **3.10+** (structural `match` statements are used throughout)
  - CUDA-capable GPU strongly recommended
  - [taming-transformers](https://github.com/CompVis/taming-transformers) — installed automatically by `setup.py`

  Key packages (see `requirements.txt`):

  ---

  ## Credits

  - [Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers) — Esser et
   al., 2021
  - [CVUSA Dataset](https://mvrl.cse.wustl.edu/datasets/cvusa/) — Zhai et al., 2017