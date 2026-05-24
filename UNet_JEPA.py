import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class CrossAttentionBridge(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, q, kv):
        # Queries define the output length, Keys/Values provide the context
        attn_out, _ = self.mha(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        return q + attn_out

class UNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.conv2(self.act(self.conv1(x)))))

class MiniUNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # A lightweight 1-level U-Net
        self.enc = UNetBlock(dim)
        self.down = nn.MaxPool2d(2)
        self.bottleneck = UNetBlock(dim)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec = UNetBlock(dim)

    def forward(self, x):
        skip = self.enc(x)
        x = self.down(skip)
        x = self.bottleneck(x)
        x = self.up(x)
        return self.dec(x + skip)

class UNetJEPA_Encoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=192, is_target=False):
        super().__init__()
        self.is_target = is_target
        self.embed_dim = embed_dim
        self.num_patches_1d = img_size // patch_size
        self.num_patches = self.num_patches_1d ** 2
        
        self.patch_embed = PatchEmbedding(patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.cnn = MiniUNet(embed_dim)

        if not is_target:
            # Cross-Attention Bridge for Context Encoder (e.g., compresses to an 8x8 grid)
            self.bottleneck_grid = 8 
            self.latent_queries = nn.Parameter(torch.randn(1, self.bottleneck_grid**2, embed_dim))
            self.bridge_in = CrossAttentionBridge(embed_dim)
            self.bridge_out = CrossAttentionBridge(embed_dim)

    def forward(self, x, masks=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed

        if self.is_target:
            B, N, C = x.shape
            H = W = int(N**0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.cnn(x)
            return x.flatten(2).transpose(1, 2)
            
        else:
            assert masks is not None, "Context Encoder requires masks"
            
            # Extract only visible patches
            visible_x = self._apply_masks(x, masks) # [B * num_masks, N_keep, dim]
            B_star = visible_x.size(0)

            # Bridge In: Variable N -> Fixed 64
            queries = self.latent_queries.expand(B_star, -1, -1)
            latent_seq = self.bridge_in(queries, visible_x)

            # Process in 2D CNN (8x8)
            C = self.embed_dim
            H = W = self.bottleneck_grid
            latent_grid = latent_seq.transpose(1, 2).reshape(B_star, C, H, W)
            latent_grid = self.cnn(latent_grid)
            latent_seq = latent_grid.flatten(2).transpose(1, 2)

            # Bridge Out: Fixed 64 -> Variable N
            out_x = self.bridge_out(visible_x, latent_seq)
            return out_x

    def _apply_masks(self, x, masks):
        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x.append(torch.gather(x, dim=1, index=mask_keep.long()))
        return torch.cat(all_x, dim=0)

class UNetJEPA_Predictor(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=192, pred_dim=96):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.pred_dim = pred_dim
        
        self.proj_in = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, pred_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, pred_dim))
        
        self.cnn = MiniUNet(pred_dim)
        self.proj_out = nn.Linear(pred_dim, embed_dim)

    def forward(self, context_x, masks_enc, masks_pred):
        B_star = context_x.size(0)
        context_x = self.proj_in(context_x)

        # Reassemble the full 14x14 grid
        full_grid = self.mask_token.expand(B_star, self.num_patches, -1).clone()
        full_grid = full_grid + self.pos_embed

        # Scatter the updated context patches back to their original coordinates
        mask_idx = torch.cat(masks_enc, dim=0) # [B_star, N_keep]
        mask_idx_expanded = mask_idx.unsqueeze(-1).expand(-1, -1, self.pred_dim)
        full_grid.scatter_(1, mask_idx_expanded.long(), context_x)

        # Run Predictor U-Net (14x14)
        C = self.pred_dim
        H = W = int(self.num_patches**0.5)
        grid_2d = full_grid.transpose(1, 2).reshape(B_star, C, H, W)
        out_2d = self.cnn(grid_2d)
        out = out_2d.flatten(2).transpose(1, 2)

        out = self.proj_out(out)

        # Gather and return ONLY the predicted target masks
        return self._extract_pred_masks(out, masks_pred, num_enc=len(masks_enc))

    def _extract_pred_masks(self, x, masks_pred, num_enc):
        all_x = []
        for m in masks_pred:
            # Repeat the mask to match the duplicated batch size from masks_enc
            m_rep = m.repeat(num_enc, 1) 
            m_keep = m_rep.unsqueeze(-1).expand(-1, -1, x.size(-1))
            all_x.append(torch.gather(x, dim=1, index=m_keep.long()))
        return torch.cat(all_x, dim=0)

    
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
                
        return self.head(features)