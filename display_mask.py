import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from Dataset import get_dataloader

def main():
    img_size = 224
    patch_size = 16
    h_patches, w_patches = img_size // patch_size, img_size // patch_size

    mask_params = {
        "patch_size": patch_size,
        "enc_mask_scale": (0.85, 1.0),
        "pred_mask_scale": (0.15, 0.2),
        "aspect_ratio": (0.75, 1.5),
        "num_enc_masks": 1,
        "num_pred_masks": 4,
        "min_keep": 10,
        "allow_overlap": False
    }

    data_loader = get_dataloader(batch_size=128, img_size=img_size, mask_params=mask_params)
    imgs, masks_enc, masks_pred = next(iter(data_loader))
    
    rand_indices = torch.randperm(imgs.size(0))[:25].tolist()
    
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    imgs = inv_normalize(imgs).clamp(0, 1)

    vis_imgs = []
    for idx in rand_indices:
        img = imgs[idx]
        
        pred_indices = masks_pred[0][idx]
        pred_mask = torch.zeros(h_patches * w_patches, dtype=torch.float32)
        pred_mask[pred_indices] = 1.0
        pred_mask = pred_mask.view(1, h_patches, w_patches)
        pred_mask = torch.nn.functional.interpolate(
            pred_mask.unsqueeze(0), size=(img_size, img_size), mode='nearest'
        ).squeeze(0)
        
        enc_indices = masks_enc[0][idx]
        enc_mask = torch.zeros(h_patches * w_patches, dtype=torch.float32)
        enc_mask[enc_indices] = 1.0
        enc_mask = enc_mask.view(1, h_patches, w_patches)
        enc_mask = torch.nn.functional.interpolate(
            enc_mask.unsqueeze(0), size=(img_size, img_size), mode='nearest'
        ).squeeze(0)

        blended_img = img.clone()
        blended_img = torch.where(pred_mask == 1, blended_img * 0.5, blended_img)
        blended_img = torch.where(enc_mask == 1, blended_img * 0.2, blended_img)
        
        vis_imgs.append(blended_img)

    grid = make_grid(torch.stack(vis_imgs), nrow=5, padding=2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig('masked_images_grid2.png', bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    main()