import torch.nn as nn
import torch
import numpy as np  
import math
from torch.nn.init import trunc_normal_


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
            self, img_size=224, patch_size=16, embed_dim=192, num_heads=3, 
            depth=3, is_predictor=False, predictor_embed_dim=None, 
            predictor_depth=None, dropout=0.1
        ):
        super().__init__()

        self.is_predictor = is_predictor
        self.predictor_embed_dim = predictor_embed_dim or embed_dim  
        self.predictor_depth = predictor_depth or depth 

        
        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2
        grid_size = int(num_patches**0.5)

        if not is_predictor:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
            pos_embed = self.__get_2d_sincos_pos_embed(embed_dim, grid_size)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            self.pos_drop = nn.Dropout(p=dropout)
        else:
            self.predictor_embed = nn.Linear(embed_dim, self.predictor_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.predictor_embed_dim))
            
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.predictor_embed_dim), requires_grad=False
            )

            pred_pos_embed = self.__get_2d_sincos_pos_embed(self.predictor_embed_dim, grid_size)
            self.predictor_pos_embed.data.copy_(torch.from_numpy(pred_pos_embed).float().unsqueeze(0))

        block_dim = self.predictor_embed_dim if is_predictor else embed_dim
        block_depth = self.predictor_depth if is_predictor else depth
        
        self.blocks = nn.ModuleList([
            TransformerEncoder(block_dim, num_heads, dropout=dropout) 
            for _ in range(block_depth)
        ])


        if not is_predictor:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.predictor_norm = nn.LayerNorm(self.predictor_embed_dim)
            self.predictor_proj = nn.Linear(self.predictor_embed_dim, embed_dim, bias=True)

        self.init_std = 0.02
        self.apply(self._init_weights)
        self.fix_init_weight()
        
        if is_predictor:
            trunc_normal_(self.mask_token, std=self.init_std)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                trunc_normal_(m.in_proj_weight, std=self.init_std)
            if m.q_proj_weight is not None:
                trunc_normal_(m.q_proj_weight, std=self.init_std)
            if m.k_proj_weight is not None:
                trunc_normal_(m.k_proj_weight, std=self.init_std)
            if m.v_proj_weight is not None:
                trunc_normal_(m.v_proj_weight, std=self.init_std)
            if m.in_proj_bias is not None:
                nn.init.constant_(m.in_proj_bias, 0)

    def fix_init_weight(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.out_proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward(self, x, masks_x=None, masks=None):
        if self.is_predictor:
            return self._forward_predictor(x, masks_x, masks)
        else:
            return self._forward_encoder(x, masks_x)
    
    def _forward_encoder(self, x, masks_x=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        if masks_x is not None:
            x = self._apply_masks(x, masks_x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.norm(x)
    
    def _forward_predictor(self, x, masks_x, masks):
        assert masks is not None and masks_x is not None, "Predictor mode requires masks_x and masks"
        
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]
        
        B = len(x) // len(masks_x)  

        x = self.predictor_embed(x)  # [B, N_ctxt, predictor_embed_dim]
        
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x_pos_embed = self._apply_masks(x_pos_embed, masks_x)
        x = x + x_pos_embed 
        

        target_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        target_pos_embed = self._apply_masks(target_pos_embed, masks)
        target_pos_embed = self._repeat_interleave_batch(target_pos_embed, B, repeat=len(masks_x))
        

        pred_tokens = self.mask_token.repeat(target_pos_embed.size(0), target_pos_embed.size(1), 1)
        pred_tokens = pred_tokens + target_pos_embed  # [B_total, N_mask, pred_dim]
        
        x = x.repeat(len(masks), 1, 1)  # [B_total, N_ctxt, pred_dim]
        
        x = torch.cat([x, pred_tokens], dim=1)  # [B_total, N_ctxt + N_mask, pred_dim]
        for block in self.blocks:
            x = block(x)
            
        x = self.predictor_norm(x)
        

        N_ctxt = x.shape[1] - pred_tokens.shape[1]
        x = x[:, N_ctxt:]  # [B_total, N_mask, pred_dim]
        
        x = self.predictor_proj(x)  # [B_total, N_mask, embed_dim]

        return x
    
    def _apply_masks(self, x, masks):
        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)
    
    def _repeat_interleave_batch(self, x, B, repeat):
        N = len(x) // B
        x = torch.cat([
            torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ], dim=0)
        return x
    
    def __get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        grid_h = np.arange(grid_size, dtype=float)
        grid_w = np.arange(grid_size, dtype=float)
        grid = np.meshgrid(grid_w, grid_h) 
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.__get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

        return pos_embed

    def __get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        emb_h = self.__get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.__get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

        return emb

    def __get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        assert embed_dim % 2 == 0

        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  

        pos = pos.reshape(-1)  
        out = np.einsum('m,d->md', pos, omega)  

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  
        
        return emb
    
    

def ViT_TinyL(img_size, patch_size, embed_dim, num_heads, dropout) : 
    return ViT(
        img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,num_heads=num_heads,
        depth=12, dropout=dropout
    )

def ViT_Predictor(img_size, patch_size, embed_dim, num_heads, dropout, predictor_embed_dim): 
    return ViT(
        img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,num_heads=num_heads,
        depth=6, dropout=dropout, is_predictor=True, predictor_embed_dim=predictor_embed_dim
    )