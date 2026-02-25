import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from fixermodule import top_k_top_p_filtering


# Convert tensors back to PIL images for display
def tensor_to_displayable(tensor):
        # Convert tensor to displayable format [0,1]
        img = ((tensor.squeeze(0) + 1) / 2).cpu()
        return img.permute(1, 2, 0).clamp(0, 1)


def single_image_inference(model, ground_image_path, device='cpu', temperature=1.0, top_k=600, top_p=0.92, save_image=False, nameadd=""):

    
    # Load and preprocess the ground image
    ground_pil = Image.open(ground_image_path).convert('RGB')
    
    # Use same transform as your training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),                              
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    ground_tensor = transform(ground_pil).unsqueeze(0).to(device)  # Add batch dimension
    
    model.eval() #Set evaluation mode
    
    with torch.no_grad():
        # Encode ground image to conditioning tokens
        ground_quant_c, _, ground_info = model.cond_stage_model.encode(ground_tensor)
        ground_indices = ground_info[2]
        
        # Handle reshape only if needed (like your training code), to get a 1D stream
        batch_size = ground_tensor.shape[0]
        if ground_indices.dim() == 1:
            ground_tokens_per_image = ground_indices.shape[0] // batch_size  
            ground_indices = ground_indices.view(batch_size, ground_tokens_per_image)
        
        # Generate satellite tokens autoregressively
        sequence = ground_indices  # Start with conditioning
        satellite_seq_length = 256  # 16x16 tokens
        
        for i in range(satellite_seq_length):
            logits, _ = model.transformer(sequence)                 #Generation 
            # logits shape: (1, seq_len, 16384) → [:, -1, :] takes only the last position → shape (1, 16384)
            # each value represents the raw "preference" of the model for each of the 16384 possible tokens

            # dividing by temperature changes how spread apart these preferences are:
            # temperature < 1.0 → differences amplified → most probable token dominates → more deterministic output
            # temperature = 1.0 → distribution unchanged
            # temperature > 1.0 → differences reduced → flatter distribution → more random/varied output
            next_token_logits = logits[:, -1, :] / temperature      
            # Rewritten top_k_top_p_filtering function
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # Sample instead of argmax, from the distribution of probabilities
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            sequence = torch.cat([sequence, next_token], dim=1)
        
        # Extract generated satellite tokens (from the sequence which is 256 ground contitioning + 256 generated)
        generated_tokens = sequence[:, -satellite_seq_length:]
        
        # Decode using first_stage_model
        h = w = 16
        z_indices_spatial = generated_tokens.view(batch_size, h, w)                         #get a view reorganized
        
        # Get quantized features from codebook
        quant_z = model.first_stage_model.quantize.embedding(z_indices_spatial)             #obtain the latent space vector of the codebook equivalent
        quant_z = quant_z.permute(0, 3, 1, 2).contiguous()                                  # [batch, embed_dim, h, w]
        
        generated_satellite_tensor = model.first_stage_model.decode(quant_z)                #decoded

    
    # Convert to displayable format
    ground_display = tensor_to_displayable(ground_tensor)                                   #
    generated_display = tensor_to_displayable(generated_satellite_tensor)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show INPUT
    axes[0].imshow(ground_display)
    axes[0].set_title("INPUT\n(Ground View)", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Show OUTPUT
    axes[1].imshow(generated_display)
    axes[1].set_title("OUTPUT\n(Generated Satellite)", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Convert back to PIL for return
    to_pil = transforms.ToPILImage()
    generated_satellite_pil = to_pil(generated_display.permute(2, 0, 1))

    #Create unique filename based on input image and parameters
    if save_image:
        # Extract filename from path (without extension)
        input_filename = os.path.splitext(os.path.basename(ground_image_path))[0]
        
        # Create unique filename with parameters
        unique_filename = f"generated_{input_filename}_{nameadd}_temp{temperature}_k{top_k}_p{top_p}.png"
        
        generated_satellite_pil.save(unique_filename)
        print(f"✅ Generated image saved as '{unique_filename}'")
    
    return generated_satellite_pil, ground_pil
