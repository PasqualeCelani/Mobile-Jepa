import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import math


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


class EmbeddingPredictor(nn.Module):
    """
    ViT-style predictor that operates on pre-computed patch embeddings.
    
    Input:  [B, N, embed_dim]  (CNN outputs + positional encoding already added)
    Output: [B, N, embed_dim]  (predicted embeddings for all positions)
    
    Used for: Masked embedding prediction in latent space (I-JEPA variant)
    """
    def __init__(
        self, 
        num_patches,      # N = number of patches (e.g., 16 for 56x56 patches)
        embed_dim,        # CNN output dimension (e.g., 192)
        predictor_embed_dim,  # Internal predictor dimension (e.g., 96)
        num_heads=3, 
        depth=12, 
        dropout=0.1,
        mlp_ratio=4.0
    ):
        super().__init__()
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        
        # Project input embeddings to predictor dimension (if different)
        if embed_dim != predictor_embed_dim:
            self.input_proj = nn.Linear(embed_dim, predictor_embed_dim)
        else:
            self.input_proj = nn.Identity()
            
        # Learnable position embeddings: [1, N, predictor_embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim))
        
        # Learnable mask token: [1, 1, predictor_embed_dim]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer blocks (operating in predictor_embed_dim space)
        self.blocks = nn.ModuleList([
            TransformerEncoder(predictor_embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Output projection back to original embed_dim (for loss comparison)
        if predictor_embed_dim != embed_dim:
            self.output_proj = nn.Linear(predictor_embed_dim, embed_dim)
        else:
            self.output_proj = nn.Identity()
            
        self.norm = nn.LayerNorm(predictor_embed_dim)
        
        # Initialize weights
        self.init_std = 0.02
        self._init_weights()
        self.fix_init_weight()
    
    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, embed_dim] - CNN patch embeddings (with positions already added)
            mask: [B, N] boolean tensor - True=keep, False=mask (optional)
                  If provided, masked positions will be replaced with mask_token
        
        Returns:
            predictions: [B, N, embed_dim] - predicted embeddings for all positions
        """
        # Project to predictor dimension
        x = self.input_proj(x)  # [B, N, predictor_embed_dim]
        
        # Apply masking if provided
        if mask is not None:
            # mask: [B, N] --> [B, N, 1] for broadcasting
            mask_expanded = mask.unsqueeze(-1)
            # Where mask=False, replace with mask_token
            x = torch.where(
                mask_expanded, 
                x, 
                self.mask_token.expand_as(x)
            )
        
        # Add positional encoding (always kept, never masked)
        x = self.pos_drop(x + self.pos_embed)
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Normalize and project back to original embed_dim
        x = self.norm(x)
        predictions = self.output_proj(x)  # [B, N, embed_dim]
        
        return predictions
    
def ViT_Predictor(num_patches, embed_dim, predictor_embed_dim, num_heads, dropout, depth=12):
    return EmbeddingPredictor(
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        num_heads=num_heads,
        depth=depth,
        dropout=dropout
    )