import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [Batch, 3, 224, 224] -> [Batch, embed_dim, 14, 14]
        x = self.proj(x)
        # Flatten and transpose: [Batch, 196, embed_dim]
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
    def __init__(self, img_size=224, patch_size=16, num_classes=100, embed_dim=768, num_heads=12, depth=3):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2
        
        # [class] Token: [1, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable position embeddings: [1, num_patches + 1, embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) # Classifier head

    def forward(self, x):
        x = self.patch_embed(x)
        
        # Prepend CLS token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = self.pos_drop(x + self.pos_embed)
        
        for block in self.blocks:
            x = block(x)
        
        # Final Norm, it takes the CLS token for the classification head
        x = self.norm(x)
        return self.head(x[:, 0])

def ViT-TinyS() : return ViT(depth=3)
def ViT-TinyM() : return ViT(depth=6)
def ViT-TinyL() : return ViT(depth=9)