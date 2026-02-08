import os
import urllib.request

for filename, url in model_urls.items():
    filepath = f'/kaggle/working/models/{filename}'
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")


def download_taming_vqgan(version=16, kaggle_flag=False):
    # Create models directory
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
    
    for filename, url in model_urls.items():
        filepath = f'/kaggle/working/models/{filename}'
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists")
        
    return
