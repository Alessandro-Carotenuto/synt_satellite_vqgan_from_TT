import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from fixermodule import top_k_top_p_filtering
from CVUSA_Manager import get_standard_transform
import csv
import config



# Convert tensors back to PIL images for display
def tensor_to_displayable(tensor):
        # Convert tensor to displayable format [0,1]
        img = ((tensor.squeeze(0) + 1) / 2).cpu()
        return img.permute(1, 2, 0).clamp(0, 1)

# Inference for one image
def single_image_inference(model, ground_image_path, real_polar_path=None, device='cpu', temperature=1.0, top_k=600, top_p=0.92, save_image=False, nameadd=""):
    # Carica e preprocessa l'immagine ground
    ground_pil = Image.open(ground_image_path).convert('RGB')
    transform = get_standard_transform()
    ground_tensor = transform(ground_pil).unsqueeze(0).to(device)
    
    # Se passiamo il percorso della polare reale, carichiamola per il confronto
    real_polar_display = None
    if real_polar_path:
        real_polar_pil = Image.open(real_polar_path).convert('RGB')
        # Applichiamo lo stesso transform per coerenza visiva (resize e normalizzazione)
        real_polar_tensor = transform(real_polar_pil).unsqueeze(0).to(device)
        real_polar_display = tensor_to_displayable(real_polar_tensor)

    model.eval()
    with torch.no_grad():
        # Encode ground image to conditioning tokens
        ground_quant_c, _, ground_info = model.cond_stage_model.encode(ground_tensor)
        ground_indices = ground_info[2]
        
        batch_size = ground_tensor.shape[0]
        if ground_indices.dim() == 1:
            ground_tokens_per_image = ground_indices.shape[0] // batch_size  
            ground_indices = ground_indices.view(batch_size, ground_tokens_per_image)
        
        # Generazione autoregressiva
        sequence = ground_indices
        satellite_seq_length = 256
        
        for i in range(satellite_seq_length):
            logits, _ = model.transformer(sequence)
            next_token_logits = logits[:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            sequence = torch.cat([sequence, next_token], dim=1)
        
        generated_tokens = sequence[:, -satellite_seq_length:]
        
        # Decode
        h = w = 16
        z_indices_spatial = generated_tokens.view(batch_size, h, w)
        quant_z = model.first_stage_model.quantize.embedding(z_indices_spatial)
        quant_z = quant_z.permute(0, 3, 1, 2).contiguous()
        generated_satellite_tensor = model.first_stage_model.decode(quant_z)

    # Preparazione display
    ground_display = tensor_to_displayable(ground_tensor)
    generated_display = tensor_to_displayable(generated_satellite_tensor)
    
    # Visualizzazione (3 colonne se abbiamo la reale, altrimenti 2)
    n_cols = 3 if real_polar_display is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    
    # 1. INPUT
    axes[0].imshow(ground_display)
    axes[0].set_title("INPUT\n(Ground View)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    if real_polar_display is not None:
        # 2. TARGET REALE (Centro)
        axes[1].imshow(real_polar_display)
        axes[1].set_title("TARGET REALE\n(Polar Satellite)", fontsize=14, fontweight='bold', color='green')
        axes[1].axis('off')
        
        # 3. OUTPUT GENERATO (Destra)
        axes[2].imshow(generated_display)
        axes[2].set_title("OUTPUT\n(Generated)", fontsize=14, fontweight='bold', color='blue')
        axes[2].axis('off')
    else:
        # 2. OUTPUT GENERATO (Destra) se manca la reale
        axes[1].imshow(generated_display)
        axes[1].set_title("OUTPUT\n(Generated)", fontsize=14, fontweight='bold', color='blue')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    to_pil = transforms.ToPILImage()
    generated_satellite_pil = to_pil(generated_display.permute(2, 0, 1))
    
    return generated_satellite_pil, ground_pil

# Testing the inference
def test_inference(model, data_root, device='cpu'):
    """Run inference on the first 5 images of the val set"""
    
    if config.OLD_SUBSET:
        csv_path = os.path.join(data_root, "val-19zl_fixed.csv")
    else:
        csv_path = os.path.join(data_root, "val.csv")
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = list(reader)[:5]
    
    for idx, row in enumerate(rows):
        ground_path = os.path.join(data_root, row[1])
        print(f"\n--- Image {idx + 1}: {os.path.basename(ground_path)} ---")
        single_image_inference(model, ground_path, device=device)
    """Run inference on the first 5 images of the val set"""