import torch, torch.nn.functional as F
from UNet_JEPA import UNetJEPA_Encoder
from Dataset import get_linear_probe_dataloaders


def knn_eval(encoder, train_loader, val_loader, device, k=20):
    encoder.eval()
    
    def extract(loader):
        feats, labels = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                f = encoder(imgs)                    # [B, 256, 12, 12]
                f = F.normalize(f.mean(dim=[2,3]), dim=1)  # [B, 256]
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    features = 16 
    encoder = UNetJEPA_Encoder(img_size=224, features=features, is_target=True)
    
    checkpoint = torch.load("./training_results/round2/checkpoint.pth", map_location=device)
    encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
    print(f"Loaded JEPA Target Encoder weights from epoch {checkpoint['epoch']}")

    encoder.to(device)

    train_loader, val_loader = get_linear_probe_dataloaders(batch_size=256)

    knn_eval(encoder, train_loader, val_loader, device)
    
    

if __name__ == "__main__":
    main()