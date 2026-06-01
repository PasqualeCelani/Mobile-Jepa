import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from Dataset import get_dataloader

def main():
    img_size = 224
    

    mask_params = {
        "patch_size": 16, 
        "enc_mask_scale": (0.40, 0.50),
        "pred_mask_scale": (0.05, 0.08),
        "aspect_ratio": (0.75, 1.5),
        "num_enc_masks": 1,
        "num_pred_masks": 6,
        "min_keep": 10,
        "allow_overlap": False
    }

    print("Loading dataloader...")
    data_loader = get_dataloader(batch_size=32, img_size=img_size, mask_params=mask_params)
    

    imgs, masks_enc, masks_pred = next(iter(data_loader))
    

    num_vis = 25
    rand_indices = torch.randperm(imgs.size(0))[:num_vis].tolist()
    
  
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    imgs = inv_normalize(imgs).clamp(0, 1)

    vis_imgs = []
    for idx in rand_indices:
        img = imgs[idx].clone() # [3, 224, 224]
        
        # [B, npred, 4] (y, x, h, w)
        pred_boxes = masks_pred[idx] 
        # [B, nenc, 4] (y, x, h, w)
        enc_boxes = masks_enc[idx]
        

        pred_mask = torch.zeros((img_size, img_size), dtype=torch.bool)
        enc_mask = torch.zeros((img_size, img_size), dtype=torch.bool)
        

        for i in range(pred_boxes.shape[0]):
            y, x, h, w = pred_boxes[i].int().tolist()
            pred_mask[y:y+h, x:x+w] = True
            

        for i in range(enc_boxes.shape[0]):
            y, x, h, w = enc_boxes[i].int().tolist()
            enc_mask[y:y+h, x:x+w] = True
            

        blended_img = img.clone()
        
        # colors [R, G, B], [3, 1]
        red_vals = torch.tensor([0.8, 0.2, 0.2]).view(3, 1)
        blue_vals = torch.tensor([0.2, 0.2, 0.8]).view(3, 1)
        
        blended_img[:, pred_mask] = blended_img[:, pred_mask] * 0.4 + red_vals * 0.6
        blended_img[:, enc_mask] = blended_img[:, enc_mask] * 0.4 + blue_vals * 0.6
        
        vis_imgs.append(blended_img.clamp(0, 1))

    grid = make_grid(torch.stack(vis_imgs), nrow=5, padding=2)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title("U-JEPA masks: RED = prediction target | BLUE = context encoder", fontsize=16)
    plt.savefig('masked_image_grid.png', bbox_inches='tight', pad_inches=0.1)
    print("Saved to masked_image_grid.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()