import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))


from models.BackbonesCNN import *


class MobileJEPA_Encoder(nn.Module):
    def __init__(self, img_size=224, features=64, is_target=False):
        super().__init__()

        self.is_target = is_target
        self.cnn = CNN_Encoder(in_channels=3, features=features)

    def forward(self, x, masks=None):
        if self.is_target:
            return self.cnn(x) 
        
        assert masks is not None, "Context Encoder requires bounding box masks"

        crops = self._extract_crops(x, masks)

        return self.cnn(crops) 

    def _extract_crops(self, imgs, masks):
        B, nenc, _ = masks.shape
        C = imgs.shape[1]

        masks_long = masks.long()
        h, w = masks_long[0, 0, 2].item(), masks_long[0, 0, 3].item()
        y, x = masks_long[:, :, 0], masks_long[:, :, 1]

        dy, dx = torch.meshgrid(torch.arange(h, device=imgs.device),
                                torch.arange(w, device=imgs.device), indexing='ij')

        b_idx = torch.arange(B, device=imgs.device).view(B, 1, 1, 1, 1)
        c_idx = torch.arange(C, device=imgs.device).view(1, 1, C, 1, 1)
        y_idx = y.view(B, nenc, 1, 1, 1) + dy.view(1, 1, 1, h, w)
        x_idx = x.view(B, nenc, 1, 1, 1) + dx.view(1, 1, 1, h, w)

        crops = imgs[b_idx, c_idx, y_idx, x_idx]
        
        return crops.view(B * nenc, C, h, w)



class MobileJEPA_Predictor(nn.Module):
    def __init__(self, img_size=224, features=64, patch_size=16, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.bottleneck_size = img_size // self.patch_size  
        self.embed_dim = embed_dim
        
        # Learnable token
        self.mask_token = nn.Parameter(torch.randn(1, self.embed_dim, 1, 1))

        nn.init.normal_(self.mask_token, std=0.02)
        
        self.cnn = PredictorNet(in_channels=self.embed_dim, features=self.embed_dim)

    def forward(self, context_feats, target_feats, masks_enc, masks_pred):
        B, C = target_feats.shape[0], target_feats.shape[1]
        nenc, npred = masks_enc.shape[1], masks_pred.shape[1]
        
        canvas = self.mask_token.expand(B, -1, self.bottleneck_size, self.bottleneck_size).clone()
        
        masks_e = (masks_enc // self.patch_size).long()
        
        h_e, w_e = masks_e[0, 0, 2].item(), masks_e[0, 0, 3].item()
        y_e, x_e = masks_e[:, :, 0], masks_e[:, :, 1]

        dy_e, dx_e = torch.meshgrid(torch.arange(h_e, device=canvas.device),
                                    torch.arange(w_e, device=canvas.device), indexing='ij')

        # Broadcast dimensions to shape [B, nenc, C, h_e, w_e]
        b_idx_e = torch.arange(B, device=canvas.device).view(B, 1, 1, 1, 1)
        c_idx = torch.arange(C, device=canvas.device).view(1, 1, C, 1, 1)
        y_idx_e = y_e.view(B, nenc, 1, 1, 1) + dy_e.view(1, 1, 1, h_e, w_e)
        x_idx_e = x_e.view(B, nenc, 1, 1, 1) + dx_e.view(1, 1, 1, h_e, w_e)

        context_feats = context_feats.view(B, nenc, self.embed_dim, h_e, w_e)
        
        canvas[b_idx_e, c_idx, y_idx_e, x_idx_e] = context_feats
        
        predicted_full_map = self.cnn(canvas)
        
        masks_p = (masks_pred // self.patch_size).long()
        h_p, w_p = masks_p[0, 0, 2].item(), masks_p[0, 0, 3].item()
        y_p, x_p = masks_p[:, :, 0], masks_p[:, :, 1]

        dy_p, dx_p = torch.meshgrid(torch.arange(h_p, device=canvas.device),
                                    torch.arange(w_p, device=canvas.device), indexing='ij')

        b_idx_p = torch.arange(B, device=canvas.device).view(B, 1, 1, 1, 1)
        y_idx_p = y_p.view(B, npred, 1, 1, 1) + dy_p.view(1, 1, 1, h_p, w_p)
        x_idx_p = x_p.view(B, npred, 1, 1, 1) + dx_p.view(1, 1, 1, h_p, w_p)
        
        pred_blocks = predicted_full_map[b_idx_p, c_idx, y_idx_p, x_idx_p].view(B * npred, C, h_p, w_p)
        target_blocks = target_feats[b_idx_p, c_idx, y_idx_p, x_idx_p].view(B * npred, C, h_p, w_p)
        
        return pred_blocks, target_blocks
    
class LinearProbeJEPA(nn.Module):
    def __init__(self, encoder, embed_dim=192, num_classes=100):
        super().__init__()
        self.encoder = encoder
        
        for param in self.encoder.parameters():
            param.requires_grad = False
            

        self.head = nn.Linear(embed_dim, num_classes)

        self.head.weight.data.normal_(mean=0.0, std=0.01)
        self.head.bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x) 
            
            if len(features.shape) == 3:
                features = features.mean(dim=1)
            elif len(features.shape) == 4:
                features = features.mean(dim=[2, 3])

            features = F.normalize(features, p=2, dim=1)         
                
        return self.head(features)