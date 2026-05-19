import torch
import torch.nn as nn


class PatchCNNExtractor(nn.Module):
    """
    Extracts patch-level embeddings from an image using a shared-weight CNN.
    
    Pipeline:
    1. Unfold image [B, 3, 224, 224] → [B, num_patches, 3, patch_size, patch_size]
    2. Flatten → [B*num_patches, 3, patch_size, patch_size]
    3. Forward through CNN (shared weights) → [B*num_patches, cnn_output_dim]
    4. Reshape → [B, num_patches, cnn_output_dim]
    5. Optional projection → [B, num_patches, embed_dim]
    6. Add learnable positional embeddings → [B, num_patches, embed_dim]
    
    Used for: I-JEPA style latent representation extraction
    """
    
    def __init__(
        self,
        cnn_model,              
        img_size=224,           
        patch_size=56,         
        embed_dim=192,          
        cnn_output_dim=None,
        dropout=0.1
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Store CNN (weights will be shared across all patches)
        self.cnn = cnn_model
        
        # Auto-detect CNN output dimension by running a dummy forward pass
        if cnn_output_dim is None:
            dummy_input = torch.zeros(1, 3, patch_size, patch_size)
            with torch.no_grad():
                dummy_output = self.cnn(dummy_input)
            cnn_output_dim = dummy_output.shape[-1]
            print(f"Auto-detected CNN output dim: {cnn_output_dim}")

        self.cnn_output_dim = cnn_output_dim
        
        if cnn_output_dim != embed_dim:
            self.proj = nn.Linear(cnn_output_dim, embed_dim)
            print(f"Adding projection: {cnn_output_dim} -> {embed_dim}")
        else:
            self.proj = nn.Identity()
        
        # Learnable positional embeddings: [1, num_patches, embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        self._init_pos_embed()
    
    def _init_pos_embed(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def unfold_to_patches(self, x):
        """
        Convert image tensor to patch tensor.
        
        Args:
            x: [B, 3, img_size, img_size]
        
        Returns:
            patches: [B, num_patches, 3, patch_size, patch_size]
        """
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Expected {self.img_size}x{self.img_size}, got {H}x{W}"
        assert self.img_size % self.patch_size == 0, "Image size must be divisible by patch size"
        
        # Use unfold to extract non-overlapping patches
        # Output: [B, C*patch_size*patch_size, num_patches]
        patches = x.unfold(2, self.patch_size, self.patch_size)  # [B, C, H/ps, W, ps]
        patches = patches.unfold(3, self.patch_size, self.patch_size)  # [B, C, H/ps, W/ps, ps, ps]
        
        # Rearrange: [B, num_patches, C, ps, ps]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, self.num_patches, C, self.patch_size, self.patch_size)
        
        return patches
    
    def forward(self, x, return_unmasked=False):
        """
        Extract patch embeddings with positional encoding.
        
        Args:
            x: [B, 3, img_size, img_size] input image
            return_unmasked: if True, also return embeddings before masking 
                           (useful for loss computation)
        
        Returns:
            embeddings: [B, num_patches, embed_dim] with positional encoding added
            raw_embeddings (optional): [B, num_patches, embed_dim] before pos_embed
        """
        B = x.shape[0]
        
        # Unfold image into patches
        patches = self.unfold_to_patches(x)  # [B, N, 3, ps, ps]
        
        # Flatten batch and patch dims for efficient CNN forward
        patches_flat = patches.view(B * self.num_patches, 3, self.patch_size, self.patch_size)
        
        # Forward through CNN (shared weights for all patches)
        cnn_outputs = self.cnn(patches_flat)  # [B*N, cnn_output_dim]
        
        # Reshape back to [B, N, cnn_output_dim]
        cnn_outputs = cnn_outputs.view(B, self.num_patches, self.cnn_output_dim)
        
        # Project to target embed_dim if needed
        embeddings = self.proj(cnn_outputs)  # [B, N, embed_dim]
        
        # Store raw embeddings for loss computation (before pos_embed)
        raw_embeddings = embeddings.clone()
        
        # Add positional encoding (position info is NEVER masked)
        embeddings = self.pos_drop(embeddings + self.pos_embed)  # [B, N, embed_dim]
        
        if return_unmasked:
            return embeddings, raw_embeddings
        
        return embeddings