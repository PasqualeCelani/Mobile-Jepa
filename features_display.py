import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from UNet_JEPA import UNetJEPA_Encoder
from Dataset import get_linear_probe_dataloaders

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def extract_all_layer_features(image, encoder, device):
    encoder.eval()
    features_dict = {}
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)
        x1 = encoder.cnn.inc(x)
        features_dict['Layer 1 (inc)\n224x224'] = x1
        x2 = encoder.cnn.down1(x1)
        features_dict['Layer 2 (down1)\n112x112'] = x2
        x3 = encoder.cnn.down2(x2)
        features_dict['Layer 3 (down2)\n56x56'] = x3
        x4 = encoder.cnn.down3(x3)
        features_dict['Layer 4 (down3)\n28x28'] = x4
        x5 = encoder.cnn.down4(x4)
        x5 = encoder.cnn.norm(x5) 
        features_dict['Layer 5 (Bottleneck)\n14x14'] = x5
    for k, v in features_dict.items():
        features_dict[k] = v[0].cpu().numpy()
    return features_dict

def plot_all_layers(features_dict):
    max_channels = 64
    
    for layer_name, feats in features_dict.items():
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
        plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = 16 
    encoder = UNetJEPA_Encoder(img_size=224, features=features, is_target=True)
    encoder.to(device)

    checkpoint = torch.load("checkpoint.pth", map_location=device)
    encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
    print(f"Loaded JEPA Target Encoder weights from epoch {checkpoint['epoch']}")

    _, val_loader = get_linear_probe_dataloaders(batch_size=256)
    images, _ = next(iter(val_loader))

    single_image = images[0]


    features_dict = extract_all_layer_features(single_image, encoder, device)
    plot_all_layers(features_dict)

if __name__ == "__main__":
    main()