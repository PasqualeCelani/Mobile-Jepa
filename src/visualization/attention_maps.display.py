import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent  
sys.path.insert(0, str(src_path))

from models.ViT import ViT_TinyL
from data.Dataset import get_linear_probe_dataloaders, DATASET_REGISTRY
from utils.config import get_config


captured_layers_weights = {}

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t_clone = tensor.clone()
    for t, m, s in zip(t_clone, mean, std):
        t.mul_(s).add_(m)
    return t_clone

def make_hook(layer_name):
    def hook(module, input, output):
        global captured_layers_weights
        captured_layers_weights[layer_name] = output[1].detach().cpu()
    return hook

def main():
    global captured_layers_weights
    
    params = get_config("../../training_results/round8/params.json")


    img_size = params["model_params"]["img_size"][0]    
    patch_size = params["model_params"]["patch_size"]    
    embed_dim = params["model_params"]["embed_dim"]
    num_heads =  params["model_params"]["num_heads"]
    dropout = params["model_params"]["dropout"]
    dataset_name = params["training_params"]["dataset-name"]

    dataset_cfg = DATASET_REGISTRY[dataset_name]
    mean, std = dataset_cfg.normalization

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = ViT_TinyL(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
    encoder.to(device)

    checkpoint = torch.load("checkpoint.pth", map_location=device)
    encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
    encoder.eval()
    print(f"Loaded ViT Target Encoder weights from epoch {checkpoint['epoch'] + 1}")

    handles = []
    num_layers = 12
    for b_idx in range(num_layers):
        block = encoder.blocks[b_idx]
        block.attn.need_weights = True
        h = block.attn.register_forward_hook(make_hook(f"Block_{b_idx}"))
        handles.append(h)

    _, val_loader = get_linear_probe_dataloaders(batch_size=32, img_size=img_size, dataset_name=dataset_name)
    images, _ = next(iter(val_loader))

    num_cols = 5
    selected_imgs = images[:num_cols]
    num_patches_side = img_size // patch_size

    attention_maps_cache = {c: {} for c in range(num_cols)}

    for col_idx in range(num_cols):
        img_tensor = selected_imgs[col_idx]
        
        with torch.no_grad():
            _ = encoder(img_tensor.unsqueeze(0).to(device))
            
        for b_idx in range(num_layers):
            current_block_name = f"Block_{b_idx}"
            matrix = captured_layers_weights[current_block_name].squeeze(0).numpy()
            
            spatial_attention = matrix.mean(axis=0) 
            spatial_grid = spatial_attention.reshape(num_patches_side, num_patches_side)
            
            spatial_grid_resized = np.array(
                torch.nn.functional.interpolate(
                    torch.tensor(spatial_grid).unsqueeze(0).unsqueeze(0),
                    size=(img_size, img_size),
                    mode='bicubic',
                    align_corners=False
                )[0, 0]
            )

            attention_maps_cache[col_idx][b_idx] = spatial_grid_resized


    for h in handles:
        h.remove()

    num_parts = 4
    layers_per_part = 3

    for part_idx in range(num_parts):
        start_layer = part_idx * layers_per_part  
        part_blocks = list(range(start_layer, start_layer + layers_per_part)) 

        fig, axes = plt.subplots(4, num_cols, figsize=(num_cols * 2.2, 4 * 2.2), layout='constrained')
        last_heatmap_im = None

        for col_idx in range(num_cols):
            img_tensor = selected_imgs[col_idx]
            img_display = unnormalize(img_tensor, mean, std).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            
            ax_img = axes[0, col_idx]
            ax_img.imshow(img_display)
            ax_img.axis('off')
            if col_idx == 2:
                ax_img.set_title(f"ViT mean attention maps trough layers (Part {part_idx + 1}/4)", fontsize=13, weight='bold', pad=8)
            if col_idx == 0:
                ax_img.text(-10, img_size // 2, "Original", rotation=90, va='center', ha='right', fontsize=11, weight='bold')


            for inner_row_idx, b_idx in enumerate(part_blocks):
                row_pos = inner_row_idx + 1
                spatial_grid_resized = attention_maps_cache[col_idx][b_idx]

                ax_attn = axes[row_pos, col_idx]
                ax_attn.imshow(img_display)
                
                last_heatmap_im = ax_attn.imshow(spatial_grid_resized, cmap='jet', alpha=0.55)
                ax_attn.axis('off')
                
                if col_idx == 0:
                    ax_attn.text(
                        -10, img_size // 2, f"Layer {b_idx + 1}", 
                        rotation=90, va='center', ha='right', fontsize=11, weight='bold'
                    )


        if last_heatmap_im is not None:
            cbar = fig.colorbar(last_heatmap_im, ax=axes.ravel().tolist(), location='right', shrink=0.5, pad=0.03)
            cbar.ax.set_ylabel('Attention focus intensity', rotation=-90, va="bottom", fontsize=11, weight='bold', labelpad=12)
            cbar.set_ticks([spatial_grid_resized.min(), spatial_grid_resized.max()])
            cbar.set_ticklabels(['Low focus', 'High focus'], fontsize=9, weight='bold')


        filename = f"vit_attention_mean_attention_map_part_{part_idx + 1}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()