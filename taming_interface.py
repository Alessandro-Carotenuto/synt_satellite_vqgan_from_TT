import os                                   
import urllib.request
import yaml
from taming.models.vqgan import VQModel
import torch

for filename, url in model_urls.items():
    filepath = f'/kaggle/working/models/{filename}'
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")


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

