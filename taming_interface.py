from glob import glob
import os                                   
import urllib.request
import yaml
from datetime import datetime
from taming.models.vqgan import VQModel
from taming.models.cond_transformer import Net2NetTransformer
import torch
from omegaconf import OmegaConf

# CREATE CONFIG FUNCTION
def create_config(configpath):
    #THE YAML FILE HAS THE FOLLOWING STRUCTURE:
    """
    model:
        base_learning_rate: 4.5e-06             
        target: taming.models.vqgan.VQModel             #target is the class we want to instantiate,  we want to pass params to the constructor of that class
        params:
            embed_dim: 256                              #This is the dimensionality of the latent space, which should match the z_channels in the ddconfig
            n_embed: 16384                              #Vocabulary Size, which should match the vocab_size in the transformer_config
            monitor: val/rec_loss                       #This is the metric that will be monitored during training, checkpointing and early stopping, 
            double_z: false                             #it's a flag that indicates whether the latent space should be doubled in size, more complex features but also comput. cost
            z_channels: 256                             #it's the number of channels in the latent space, which should match the embed_dim parameter above
            resolution: 256                             #it's the resolution of the input images, which should be consistent with the training data of the VQ-GAN model
            in_channels: 3                              #it's the number of channels in the input images (3 for RGB), should be consistent with the train data of the VQ-GAN model
            out_ch: 3                                   #it's the number of channels in the output images (3 for RGB), same consistency requirement     
            ch: 128                                     #it's the base number of channels in the enc and dec networks, the capacity of the model, same consistency requirement
            ch_mult:                                    #it's the channel multiplier for each level of the encoder and decoder networks, which determines how the number of channels changes as we go deeper into the network, same consistency requirement
            - 1
            - 1
            - 2
            - 2
            - 4
            num_res_blocks: 2                           #it's the number of residual blocks in the encoder and decoder networks, same consistency requirement
            attn_resolutions:                           #it's the resolutions at which attention is applied in the encoder and decoder networks, same consistency requirement
            - 16
            dropout: 0.0                                #it's the dropout rate used in the encoder and decoder networks, same consistency requirement

        lossconfig:                                     #LOSSCONFIG is the configuration for the loss function used during training. 

            target: taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator                 #This would be the actual loss function used during training
                
                #Those are the parameters for the VQLPIPSWithDiscriminator loss function, which combines a perceptual loss (LPIPS) 
                # with an adversarial loss from a discriminator.

                disc_conditional: false                                                         
                disc_in_channels: 3
                disc_start: 0
                disc_weight: 0.75
                disc_num_layers: 2
                codebook_weight: 1.0
"""

    with open(configpath,'r') as yamlconfig:
        vqgan_yaml_config=yaml.safe_load(yamlconfig)

    complete_system_configuration = OmegaConf.create({
        
        # FIRST STAGE CONFIG ------------------------------------------------

        "first_stage_config": vqgan_yaml_config['model'], #Same VQ-GAN configuration for the first stage.
        
        # CONDITIONAL STAGE CONFIG --------------------------------------------
        # cond_stage_config is setting the conditional stage to use the same VQ-GAN for both images (input and target)

        "cond_stage_config": "__is_first_stage__",              # Use same VQ-GAN for both images

        # TRANSFORMER CONFIG WITH DROPOUT --------------------------------------

        "transformer_config": {
            "target": "taming.modules.transformer.mingpt.GPT",
            "params": {
                "vocab_size": 16384,  # This matches the codebook size of the VQ-GAN
                "block_size": 512,    # Sequence length (for 16x16 = 256 tokens x2)
                "n_layer": 12,        # Number of transformer layers, can be adjusted based on model capacity needs
                "n_head": 8,          # Number of attention heads, can be adjusted based on model capacity needs
                "n_embd": 512,        # Embedding dimension, can be adjusted based on model capacity needs
                # --- ADDED DROPOUT FOR REGULARIZATION ---
                "embd_pdrop": 0.2,    # Dropout on embeddings
                "resid_pdrop": 0.2,   # Dropout on residual connections
                "attn_pdrop": 0.2     # Dropout on attention weights
            }
        }
    })

    # Define device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    return complete_system_configuration, device

# VQ-GAN DOWNLOADING, LOADING AND CHECKPOINTING FUNCTIONS
def download_taming_vqgan(version=16, kaggle_flag=False):  
    if kaggle_flag:
        os.makedirs('/kaggle/working/models', exist_ok=True)
    else:
        os.makedirs('models', exist_ok=True)
    

    #16 == f16 & 16384
    if version==16:
        model_urls = {
            'vqgan_imagenet_f16_16384.yaml': 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
            'vqgan_imagenet_f16_16384.ckpt': 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
        }
    
    filepaths=[]

    for filename, url in model_urls.items():
        if kaggle_flag:
            filepath = f'/kaggle/working/models/{filename}'
        else:
            filepath = f'models/{filename}'
        
        filepaths.append(filepath)

        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists")
        
    return filepaths

def load_vqgan_model(config_path, checkpoint_path):
    # Load config with regular yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = VQModel(**config['model']['params'])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.eval() # Set model to evaluation mode, some layers like dropout or batchnorm will behave differently during inference
    #set like that cause we are not training the model, we are just using it for inference, so we want it to behave differently than during training
    return model

# MANUAL FORWARD PASS FUNCTION

def manual_forward_pass(model, satellite_imgs, ground_imgs):
    

    # THIS IS A REWRITE OF THE FORWARD PASS LOGIC FROM THE TRAINING LOOP, 
    # IT IS REWRITTEN TO FIX AN ISSUE WITH THE SHAPES OF THE ENCODED INDICES,
    # AND TO SHOW HOW TO MANUALLY PERFORM THE FORWARD PASS THROUGH THE MODEL.

    #WE ARE CALLING METHODS FROM THE Net2NetTransformer CLASS TO ENCODE THE IMAGES INTO LATENT REPRESENTATION INDICES,

    # GET RAW ENCODED INDICES AND THEN RESHAPE THEM TO THE CORRECT SHAPE FOR THE TRANSFORMER
    # ENCODING IS: WE GET LATENT SPACE, THEN WE GO THROUGH VECTOR QUANTIZATION AND MATCH THE VECTOR TO THE CLOSEST CODEBOOKS ENTRIES

    _, z_indices_raw = model.encode_to_z(satellite_imgs)    # ENCODE SATELLITE IMGs -> LATENT REPRESENTATION INDICES, WE GET RAW INDICES WITHOUT RESHAPING
    _, c_indices_raw = model.encode_to_c(ground_imgs)       # ENCODE GROUND IMGs -> LATENT REPRESENTATION INDICES, WE GET RAW INDICES WITHOUT RESHAPING
    
    # THOSE RETURN 1D VECTOR OF INDICES, WE NEED TO RESHAPE THEM TO (BATCH_SIZE, NUM_TOKENS) FOR THE TRANSFORMER

    batch_size = satellite_imgs.shape[0]                    # WE GET THE BATCH SIZE FROM THE INPUT IMAGES, ASSUMING THE FIRST DIMENSION IS THE BATCH SIZE
    z_indices = z_indices_raw.view(batch_size, -1)          # RESHAPE (BATCH_SIZE, NUM_TOKENS), -1 INFER THE SECOND DIMENSION BASED ON THE NUMBER OF ELEM
    c_indices = c_indices_raw.view(batch_size, -1)
    
    # Manual forward pass logic (from STEP-CHECKPOINT)
    cz_indices = torch.cat((c_indices, z_indices), dim=1)   # CONCATENATES TOKENS
    logits, _ = model.transformer(cz_indices[:, :-1])       # INTO THE TRANSFORMER; EXCEPT LAST TOKEN , WHICH WE WANT TO PREDICT
    logits = logits[:, c_indices.shape[1]-1:]               # SLICE OFF THE GROUND PREFIX TOKENS TO GET ONLY THE SATELLITE PART OF THE LOGITS, WHICH SHOULD MATCH THE SHAPE OF z_indices
    target = z_indices                                      # TARGET IS THE ORIGINAL z_indices, WHICH WE WANT TO PREDICT WITH THE LOGITS FROM THE TRANSFORMER
    
    return logits, target

# SAVING, LOADING AND CHECKPOINTING FUNCTIONS FOR THE TRANSFORMER MODEL (VQ-GAN IS FROZEN, SO WE ONLY SAVE THE TRANSFORMER WEIGHTS AND OPTIMIZER STATE)

def save_checkpoint(model, optimizer, epoch, loss, base_name="cvusa_ground2satellite", save_dir=None):
    """Save only the transformer weights — VQGAN is frozen and doesn't need saving"""
    
    # IF save_dir is not provided, use the current working directory
    if save_dir is None:
        save_dir = os.getcwd()
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")                    # Create a timestamp for the checkpoint filename, format: YYYYMMDD_HHMMSS
    filename = f"{base_name}_epoch{epoch}_loss{loss:.3f}_{timestamp}.pth"   # Create a unique filename using the base name, epoch, loss, and timestamp
    full_path = os.path.join(save_dir, filename)                            # Get the full path to save the checkpoint
    
    checkpoint = {
        'epoch': epoch,
        'transformer_state_dict': model.transformer.state_dict(),           # Only transformer weights
        'optimizer_state_dict': optimizer.state_dict(),                     # Optimizer state to resume training same learning rate and momentum, etc.
        'loss': loss,                                                       # Save the loss value
        'timestamp': timestamp,                                             # Save the timestamp for reference
        'transformer_config': {                                             # Save the transformer configuration for reference and reproducibility
            'vocab_size': model.transformer.config.vocab_size,
            'block_size': model.transformer.config.block_size,
            'n_layer': model.transformer.config.n_layer,
            'n_head': model.transformer.config.n_head,
            'n_embd': model.transformer.config.n_embd,
        }
    }
    
    torch.save(checkpoint, full_path)                                       # Save the checkpoint to the specified path
    print(f"✅ Checkpoint saved: {full_path}")                              
    return full_path

def load_saved_model(checkpoint_path, vqgan_checkpoint_path=None, kaggle_flag=False):
    """
    Rebuild the model the same way train_transformer.py does,
    then load the saved transformer weights on top.
    """
    
    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load the checkpoint with weights_only=False to get the full checkpoint including epoch and loss information
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)                   
    print(f"Checkpoint loaded — epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")
    
    # Get VQGAN — download if not provided, from the hardcoded path
    if vqgan_checkpoint_path is None:
        print("No VQGAN path provided, downloading...")
        [configpath, vqgan_checkpoint_path] = download_taming_vqgan(version=16, kaggle_flag=kaggle_flag)
    else:
        # Still need the config path
        [configpath, _] = download_taming_vqgan(version=16, kaggle_flag=kaggle_flag)
    
    # Rebuild model — same as train_transformer.py
    fullsystem_config, device = create_config(configpath)                   # Create the complete system config and choose device
    fullsystem_config_no_ckpt = fullsystem_config.copy()                    # Create a copy of the full config to modify
    fullsystem_config_no_ckpt.first_stage_config.params.ckpt_path = None    # Set the checkpoint path to None to prevent automatic loading by the config

    # Rebuild the model using the same configuration, but with the modified config that does not load the checkpoint automatically
    model = Net2NetTransformer(                                             
        transformer_config=fullsystem_config.transformer_config,
        first_stage_config=fullsystem_config_no_ckpt.first_stage_config,
        cond_stage_config=fullsystem_config.cond_stage_config,
        first_stage_key="satellite",
        cond_stage_key="ground",
        unconditional=False
    )

    # Load VQGAN weights and freeze
    vqgan_state = torch.load(vqgan_checkpoint_path, map_location=device, weights_only=False)
    model.first_stage_model.load_state_dict(vqgan_state["state_dict"], strict=False)
    for param in model.first_stage_model.parameters():
        param.requires_grad = False
    print("✅ VQGAN loaded and frozen from checkpoint:", vqgan_checkpoint_path)

    # Load transformer weights
    model.transformer.load_state_dict(checkpoint['transformer_state_dict'])
    print("✅ Transformer weights loaded from checkpoint:", checkpoint_path)

    # Move model to device
    model = model.to(device)
    
    return model, checkpoint, device

def load_with_optimizer(checkpoint_path, vqgan_checkpoint_path=None, kaggle_flag=False, lr=5e-4):
    """Load model with optimizer state for resuming training"""
    
    # This function is a wrapper around load_saved_model that also initializes the optimizer and loads its state from the checkpoint if available.
    model, checkpoint, device = load_saved_model(checkpoint_path, vqgan_checkpoint_path, kaggle_flag)   
    
    # Initialize optimizer for the transformer parameters only, since VQGAN is frozen and we are not updating its weights
    optimizer = torch.optim.AdamW(model.transformer.parameters(), lr=lr, betas=(0.9, 0.95))


    # Load optimizer state if available in the checkpoint, this allows us to resume training with the same learning rate schedule and momentum, etc.
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer state restored")
    else:
        print("⚠️  No optimizer state found, using fresh optimizer")
    
    return model, optimizer, checkpoint, device

def find_latest_checkpoint(base_name, save_dir=None):
    """Find the most recently modified checkpoint matching base_name"""
    
    # If save_dir is not provided, use the current working directory
    if save_dir is None:
        save_dir = os.getcwd()
    
    # Check if the directory exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Directory not found: {save_dir}")
    
    # Use glob to find all files matching the pattern, then sort by modification time to get the latest one
    matches = glob(os.path.join(save_dir, f"{base_name}*.pth"))
    
    # If no matches found, raise an error
    if not matches:
        raise FileNotFoundError(f"No checkpoints found with base name '{base_name}' in {save_dir}")
    
    # Sort by modification time, newest first
    matches.sort(key=os.path.getmtime, reverse=True)

    # Print the found checkpoints and return the latest one
    print(f"Found {len(matches)} checkpoint(s), using: {os.path.basename(matches[0])}")
    return matches[0]