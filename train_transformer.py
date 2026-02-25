import torch
from fixermodule import fix_torch_import_issue, fix_inject_top_k_p_filtering
from taming_interface import download_taming_vqgan, create_config
from taming.models.cond_transformer import Net2NetTransformer
from taming_interface import manual_forward_pass
import torch.nn.functional as F 

def train_one_epoch(model, train_dataloader, optimizer, scaler, device):
    """Train for one epoch and return average loss"""
    model.train()           #Set Training mode
    epoch_loss = 0          #Starting Loss
    num_batches = 0         #Starting Batch?
    
    for batch_idx, batch in enumerate(train_dataloader):
        # Move to device
        ground_imgs = batch['ground'].to(device)
        satellite_imgs = batch['satellite'].to(device)
        
        # Forward pass with mixed precision
        with autocast():
            logits, target = manual_forward_pass(model, satellite_imgs, ground_imgs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),label_smoothing=0.1)
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Track loss
        epoch_loss += loss.item()
        num_batches += 1
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return epoch_loss / num_batches



def main():
    fix_torch_import_issue(kaggle_flag=False)                                               # Set to True if running in Kaggle environment
    [configpath, checkpointpath] = download_taming_vqgan(version=16, kaggle_flag=False)     # Set to True if running in Kaggle environment
    #pretrained_vqgan = load_vqgan_model(configpath, checkpointpath)                         # Load the VQGAN model using the downloaded files
    fullsystem_config, device = create_config(configpath)                                   # Create the complete system config and choose device
    fix_inject_top_k_p_filtering()                                                          # Inject filtering function into transformers module to fix import
    vqgan_state = torch.load(checkpointpath,                                                # Load checkpoint with explicit weights_only=False
                        map_location=device, # Ensure the checkpoint is loaded on the correct device
                        weights_only=False)  # This is important to load the full checkpoint not just the model weights. Possible resuming training if needed.
    
    # Modify config to not load checkpoint automatically:

    # Because we are loading the checkpoint manually with the correct map_location and weights_only settings,
    # we want to prevent the config from trying to load it again with potentially incorrect settings.
    # This gives us more control over how the checkpoint is loaded and ensures it works correctly on our device.

    fullsystem_config_no_ckpt = fullsystem_config.copy()                   # Create a copy of the full config to modify      
    fullsystem_config_no_ckpt.first_stage_config.params.ckpt_path = None   # Set the checkpoint path to None to prevent automatic loading by the config

    # Create model without loading checkpoint
    # Net2Net transformer is the main model that combines the VQGAN and the transformer for training.
    # is a class from the code from the repo of Taming-transformers that we are using to create our model. 
    # It takes the configurations for the transformer, the first stage (VQGAN), and the conditioning stage, and initializes the model accordingly.
    model = Net2NetTransformer(
        transformer_config=fullsystem_config.transformer_config,            # Use the original transformer config, we are not modifying it
        first_stage_config=fullsystem_config_no_ckpt.first_stage_config,    # Modified Config, cause we are loading manually 
        cond_stage_config=fullsystem_config.cond_stage_config,              # Use the original conditioning stage config, we are not modifying it
        first_stage_key="satellite",                                        # This is the key to access VQGAN output
        cond_stage_key="ground",                                            # This is the key to access conditioning stage output   
        unconditional=False)                                                # We are not using unconditional training, we use conditioning information

    # Manually load the VQ-GAN weights
    # We load the checkpoint manually with the correct map_location and weights_only settings to ensure it works correctly on our device.
    model.first_stage_model.load_state_dict(vqgan_state["state_dict"], strict=False) 
    print("Model created and VQ-GAN weights loaded successfully!")
    del vqgan_state #clean variable of the vqgan weights

    #Freeze the VQ-GAN parameters
    # We freeze the VQ-GAN parameters because we are not training the VQ-GAN, we are only using it for inference to encode and decode images.
    for param in model.first_stage_model.parameters():  
        param.requires_grad = False # requires_grad=false means that we do not want to compute gradients for these parameters during backpropagation

    # Since cond_stage_model is the same as first_stage_model, it's already frozen, we don't need to freeze it again.
    # Otherwise we would write:
    # for param in model.cond_stage_model.parameters():  
    #     param.requires_grad = False

    # Verify the freeze worked
    vqgan_params = sum(p.numel() for p in model.first_stage_model.parameters() if p.requires_grad) 
    transformer_params = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad) 

    print(f"VQ-GAN trainable parameters: {vqgan_params}")
    print(f"Transformer trainable parameters: {transformer_params}")

    #Move model to GPU if GPU is available
    model = model.to(device)
    print("Model moved to",device)

