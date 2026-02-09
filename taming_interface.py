import os                                   
import urllib.request
import yaml
from taming.models.vqgan import VQModel
import torch
from OmegaConf import OmegaConf


def download_taming_vqgan(version=16, kaggle_flag=False):  #Download the VQGAN model files and return their file paths
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

        "first_stage_config": {vqgan_yaml_config['model']}, #Same VQ-GAN configuration for the first stage.
        
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

