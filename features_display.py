import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from UNet_JEPA import UNetJEPA_Encoder
from Dataset import get_linear_probe_dataloaders
import math

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def plot_feature_maps(target_encoder, image, device, num_channels=16):
    target_encoder.eval()
    with torch.no_grad():
        features = target_encoder(image.unsqueeze(0).to(device))
    
    # Move to CPU and remove batch dimension
    features = features[0].cpu().numpy() # Expected shape: (C, H, W) or (Num_Patches, Embed_Dim)
    
    if len(features.shape) == 1:
        raise ValueError(
            f"The encoder output is completely flattened to {features.shape}. "
            "To plot 2D feature maps, you must modify your UNetJEPA_Encoder's forward pass "
            "to return the spatial feature maps (B, C, H, W) before global pooling is applied."
        )
    
    # If the shape is (Num_Patches, Embed_Dim), reshape back to (Embed_Dim, P, P)
    if len(features.shape) == 2:
        num_patches, embed_dim = features.shape
        P = int(math.sqrt(num_patches))
        # Swap axes to make it (Embed_Dim, Num_Patches) --> (Embed_Dim, P, P)
        features = features.T.reshape(embed_dim, P, P)
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Target Encoder: Raw Feature Maps (First 16 Channels)', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < features.shape[0]:
            ax.imshow(features[i], cmap='viridis')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_pca_latents(target_encoder, image, device):
    target_encoder.eval()
    with torch.no_grad():
        features = target_encoder(image.unsqueeze(0).to(device))
        
    features = features[0].cpu().numpy()
    
    if len(features.shape) == 1:
        print("Skipping PCA: Target encoder output is a single 1D vector (pooled features).")
        return

    if len(features.shape) == 2:
        num_patches, embed_dim = features.shape
        P = int(math.sqrt(num_patches))
        features = features.T.reshape(embed_dim, P, P)
        
    C, H, W = features.shape
    features_flat = features.transpose(1, 2, 0).reshape(-1, C)
    
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_flat)
    
    pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0))
    pca_img = pca_features.reshape(H, W, 3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    img_display = unnormalize(image.clone()).permute(1, 2, 0).cpu().numpy()
    img_display = np.clip(img_display, 0, 1)
    
    ax1.imshow(img_display)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(pca_img)
    ax2.set_title("PCA of Latent Space (RGB)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    encoder = UNetJEPA_Encoder(img_size=224, patch_size=16, embed_dim=192, is_target=True)
    encoder.to(device)
        

    checkpoint = torch.load("checkpoint.pth", map_location=device)
    encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
    print(f"Loaded JEPA Target Encoder weights from epoch {checkpoint['epoch']}")

    _, val_loader = get_linear_probe_dataloaders(batch_size=256)

    images, labels = next(iter(val_loader))
    single_image = images[0] 

    plot_feature_maps(encoder, single_image, device)
    plot_pca_latents(encoder, single_image, device)

if __name__ == "__main__":
    main()