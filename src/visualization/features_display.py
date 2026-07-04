import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent  
sys.path.insert(0, str(src_path))

from models.Mobile_JEPA import MobileJEPA_Encoder
from data.Dataset import get_linear_probe_dataloaders, DATASET_REGISTRY
from utils.config import get_config

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def extract_all_layer_features(image, encoder, device):
    encoder.eval()
    features_dict = {}

    features_dict['Original'] = image.unsqueeze(0)

    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        x1 = encoder.cnn.inc(x)
        features_dict['Layer 1 (inc)'] = x1
        x2 = encoder.cnn.down1(x1)
        features_dict['Layer 2 (down1)'] = x2
        x3 = encoder.cnn.down2(x2)
        features_dict['Layer 3 (down2)'] = x3
        x4 = encoder.cnn.down3(x3)
        features_dict['Layer 4 (down3)'] = x4
        x5 = encoder.cnn.down4(x4)
        x5 = encoder.cnn.norm(x5) 
        features_dict['Layer 5 (Bottleneck)'] = x5

    for k, v in features_dict.items():
        features_dict[k] = v[0].cpu().numpy()

    return features_dict

def plot_all_layers(features_dict, mean, std):
    max_channels = 64
    
    for layer_name, feats in features_dict.items():
        if 'Original' in layer_name:
            img_tensor = torch.tensor(feats)
            img_tensor = unnormalize(img_tensor, mean, std).clamp(0, 1)
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            fig.suptitle(f'{layer_name}', fontsize=16)
            ax.imshow(img_np)
            ax.axis('off')
            plt.tight_layout()
            
            safe_filename = layer_name.replace('\n', '_').replace(' ', '_').replace('(', '').replace(')', '')
            filename = f"{safe_filename}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)

            continue

        C, _, _ = feats.shape
        num_to_show = min(C, max_channels)
        grid_size = int(np.ceil(np.sqrt(num_to_show)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
        fig.suptitle(f'{layer_name} - {num_to_show} Channels', fontsize=16)
        
        for idx, ax in enumerate(axes.flatten()):
            if idx < num_to_show:
                channel_img = feats[idx]
                channel_img = (channel_img - channel_img.min()) / (channel_img.max() - channel_img.min() + 1e-8)
                ax.imshow(channel_img, cmap='viridis')
            ax.axis('off')
        
        plt.tight_layout()
        
        safe_filename = layer_name.replace('\n', '_').replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"{safe_filename}.png"
        
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

def main():
    params = get_config("../../training_results/round10/params.json")

    img_size = params["model_params"]["img_size"][0]
    features = params["model_params"]["features"]
    dataset_name = params["training_params"]["dataset-name"]

    dataset_cfg = DATASET_REGISTRY[dataset_name]
    mean, std = dataset_cfg.normalization

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = MobileJEPA_Encoder(img_size=img_size, features=features, is_target=True)
    encoder.to(device)

    checkpoint = torch.load("checkpoint.pth", map_location=device)
    encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
    print(f"Loaded JEPA Target Encoder weights from epoch {checkpoint['epoch'] + 1}")

    _, val_loader = get_linear_probe_dataloaders(batch_size=256, img_size=img_size, dataset_name=dataset_name)
    images, _ = next(iter(val_loader))

    single_image = images[1]


    features_dict = extract_all_layer_features(single_image, encoder, device)
    plot_all_layers(features_dict, mean, std)

if __name__ == "__main__":
    main()