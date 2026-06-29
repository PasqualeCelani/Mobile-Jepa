import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import sys
import os
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent  
sys.path.insert(0, str(src_path))

from data.Dataset import get_dataloader, DATASET_REGISTRY
from utils.config import get_config

def main():
    params = get_config("../../training_results/params.json")

    img_size = params["model_params"]["img_size"][0]
    mask_params = params["mask_params"]
    dataset_name = params["training_params"]["dataset-name"]
    patch_size = params["model_params"]["patch_size"]

    is_IJepa = True

    dataset_cfg = DATASET_REGISTRY[dataset_name]
    mean, std = dataset_cfg.normalization

    print("Loading dataloader...")
    data_loader = get_dataloader(batch_size=32, img_size=img_size, mask_params=mask_params, dataset_name=dataset_name)
    

    imgs, masks_enc, masks_pred = next(iter(data_loader))
    

    num_vis = 25
    rand_indices = torch.randperm(imgs.size(0))[:num_vis].tolist()
    
  
    inv_normalize = T.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std]
    )
    imgs = inv_normalize(imgs).clamp(0, 1)

    vis_imgs = []
    for idx in rand_indices:
        img = imgs[idx].clone() # [3, 224, 224]
        
        if not is_IJepa:
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
        else:
            h_patches = img_size // patch_size
            w_patches = img_size // patch_size

            pred_mask = torch.zeros(h_patches * w_patches, dtype=torch.float32)
            enc_mask = torch.zeros(h_patches * w_patches, dtype=torch.float32)


            for i in range(len(masks_pred)):
                pred_indices = masks_pred[i][idx]
                pred_mask[pred_indices] = 1.0 

            for i in range(len(masks_enc)):
                enc_indices = masks_enc[i][idx]
                enc_mask[enc_indices] = 1.0  

            pred_mask = pred_mask.view(1, h_patches, w_patches)
            pred_mask = torch.nn.functional.interpolate(
                pred_mask.unsqueeze(0), size=(img_size, img_size), mode='nearest'
            ).squeeze(0)

            enc_mask = enc_mask.view(1, h_patches, w_patches)
            enc_mask = torch.nn.functional.interpolate(
                enc_mask.unsqueeze(0), size=(img_size, img_size), mode='nearest'
            ).squeeze(0)
            

        blended_img = img.clone()

        if not is_IJepa:
            # colors [R, G, B], [3, 1]
            red_vals = torch.tensor([0.8, 0.2, 0.2]).view(3, 1)
            blue_vals = torch.tensor([0.2, 0.2, 0.8]).view(3, 1)

            blended_img[:, pred_mask] = blended_img[:, pred_mask] * 0.4 + red_vals * 0.6
            blended_img[:, enc_mask] = blended_img[:, enc_mask] * 0.4 + blue_vals * 0.6
        else:
            # colors [R, G, B], [3, 1, 1]
            red_vals = torch.tensor([0.8, 0.2, 0.2]).view(3, 1, 1)
            blue_vals = torch.tensor([0.2, 0.2, 0.8]).view(3, 1, 1)

            blended_img = torch.where(pred_mask == 1, blended_img * 0.4 + red_vals * 0.6, blended_img)
            blended_img = torch.where(enc_mask == 1, blended_img * 0.4 + blue_vals * 0.6, blended_img)
        
        vis_imgs.append(blended_img.clamp(0, 1))

    grid = make_grid(torch.stack(vis_imgs), nrow=5, padding=2)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')

    if is_IJepa:
        plt.title("I-JEPA masks: RED = prediction target | BLUE = context encoder", fontsize=16)
    else:
        plt.title("Mobile-JEPA masks: RED = prediction target | BLUE = context encoder", fontsize=16)
        
    plt.savefig('masked_image_grid.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()