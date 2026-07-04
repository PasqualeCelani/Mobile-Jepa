import torch, torch.nn.functional as F

import sys
import os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from models.Mobile_JEPA import MobileJEPA_Encoder
from models.ViT import *
from data.Dataset import get_linear_probe_dataloaders
from utils.config import get_config


def knn_eval(encoder, train_loader, val_loader, device, is_vit_based, k=20):
    encoder.eval()
    
    def extract(loader):
        feats, labels = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                f = encoder(imgs)     

                if not is_vit_based:
                    f = F.normalize(f.mean(dim=[2,3]), dim=1)  
                else:
                    f = F.normalize(f.mean(dim=1), dim=1)

                feats.append(f.cpu())
                labels.append(lbls)
        return torch.cat(feats), torch.cat(labels)

    train_feats, train_labels = extract(train_loader)
    val_feats,   val_labels   = extract(val_loader)

    # Cosine similarity matrix
    sim = val_feats @ train_feats.T            # [N_val, N_train]
    topk = sim.topk(k, dim=1).indices         # [N_val, k]
    pred = train_labels[topk].mode(dim=1).values

    acc = (pred == val_labels).float().mean().item()
    print(f"kNN (k={k}) accuracy: {acc*100:.2f}%")
    return acc


def main():
    params = get_config("../training_results/round11/params.json")

    img_size = params["model_params"]["img_size"][0]   
    features = params["model_params"]["features"]  
    batch_size = params["test_params"]["knn"]["batch_size"]
    dataset_name = params["test_params"]["knn"]["dataset-name"]
    k = params["test_params"]["knn"]["k"]

    IS_VIT_BASED = False

    if IS_VIT_BASED:
        embed_dim = params["model_params"]["embed_dim"]
        num_heads =  params["model_params"]["num_heads"]
        predictor_embed_dim = params["model_params"]["predictor_embed_dim"]
        dropout = params["model_params"]["dropout"]
        patch_size = params["model_params"]["patch_size"]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not IS_VIT_BASED:
        encoder = MobileJEPA_Encoder(img_size=img_size, features=features, is_target=True)
    else:
        encoder = ViT_TinyL(img_size, patch_size, embed_dim, num_heads, dropout)
    
    checkpoint = torch.load("checkpoint.pth", map_location=device)
    encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
    print(f"Loaded JEPA Target Encoder weights from epoch {checkpoint['epoch']}")

    encoder.to(device)

    train_loader, val_loader = get_linear_probe_dataloaders(batch_size=batch_size, img_size=img_size, dataset_name=dataset_name)

    knn_eval(encoder, train_loader, val_loader, device, IS_VIT_BASED, k)
    
    

if __name__ == "__main__":
    main()