import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config import get_config
from Mobile_JEPA import MobileJEPA_Encoder
from Dataset import get_linear_probe_dataloaders

def main():
    params = get_config("./params.json")
    img_size = params["model_params"]["img_size"][0]   
    features = params["model_params"]["features"]  
    batch_size = params["test_params"]["knn"]["batch_size"]
    dataset_name = params["test_params"]["knn"]["dataset-name"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = MobileJEPA_Encoder(img_size=img_size, features=features, is_target=True)
    checkpoint = torch.load("checkpoint.pth", map_location=device)
    encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
    print(f"Loaded JEPA Target Encoder weights from epoch {checkpoint['epoch']}")
    
    encoder.to(device)
    encoder.eval()
    
    _, val_loader = get_linear_probe_dataloaders(
        batch_size=batch_size, 
        img_size=img_size, 
        dataset_name=dataset_name
    )
    

    all_feats = []
    all_labels = []
    
    print("Extracting features...")
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            f = encoder(imgs)                 # [B, C, H, W]
            f = f.mean(dim=[2, 3])            # Global Average Pooling -> [B, C]
            f = F.normalize(f, p=2, dim=1)    # L2 Normalization -> [B, C]
            
            all_feats.append(f.cpu().numpy())
            all_labels.append(lbls.numpy())
            
    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # This prevents t-SNE from crashing on large datasets such as imagenet, the threshold depends on the avaible RAM
    MAX_SAMPLES = 5000
    if len(all_feats) > MAX_SAMPLES:
        print(f"Subsampling from {len(all_feats)} to {MAX_SAMPLES} for faster t-SNE...")
        indices = np.random.choice(len(all_feats), MAX_SAMPLES, replace=False)
        all_feats = all_feats[indices]
        all_labels = all_labels[indices]
        

    print(f"Running 3D t-SNE on {len(all_feats)} samples...")
    perplexity = min(30.0, len(all_feats) - 1)
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity, n_iter=1000)
    feats_3d = tsne.fit_transform(all_feats)
    

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)
    

    if num_classes <= 20:
        cmap = plt.get_cmap('tab20')
    else:
        cmap = plt.get_cmap('turbo')
        

    norm = plt.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max())
    
    scatter = ax.scatter(
        feats_3d[:, 0], feats_3d[:, 1], feats_3d[:, 2],
        c=all_labels, cmap=cmap, norm=norm,
        s=15,          
        alpha=0.7,     
        edgecolors='none'
    )
    
    if num_classes <= 20:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=cmap(norm(lbl)), markersize=8, label=str(lbl))
            for lbl in unique_labels
        ]
        ax.legend(handles=handles, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label('Class Label Index', rotation=270, labelpad=20)
        
        tick_step = max(1, num_classes // 10) 
        cbar.set_ticks(np.arange(unique_labels.min(), unique_labels.max() + 1, tick_step))

    ax.set_title(f"3D Latent space t-SNE on {dataset_name}", fontsize=16, pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    save_path = "latent_space_3d_matplotlib.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()