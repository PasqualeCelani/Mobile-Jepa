import torch.nn as nn


class Stem(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio = 2.0):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)
        self.use_res = in_channels == out_channels
        
        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return out + x if self.use_res else out



class LeanSHSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim, bias=False)  # Represents W_q @ W_k^T
        self.w2 = nn.Linear(dim, dim, bias=False)  # Represents W_v @ W_o
        self.scale = dim ** -0.5

    def forward(self, x):
        # x shape: (B, N, C)
        attn = (self.w1(x) @ x.transpose(1, 2)) * self.scale
        attn = attn.softmax(dim=-1)
        return attn @ self.w2(x)


class SEBlock(nn.Module):
    def __init__(self, dim, reduction = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x).flatten(1)
        y = self.fc(y).unsqueeze(-1).unsqueeze(-1)
        return x * y



class SpatialBlock(nn.Module):
    def __init__(self, dim, mlp_ratio = 2.0):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = LeanSHSA(dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm2 = nn.BatchNorm2d(dim)
        
        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        x_attn = self.attn(self.norm1(x).flatten(2).transpose(1, 2))
        x_attn = x_attn.transpose(1, 2).reshape(B, C, H, W)
        x = x + x_attn
        
        x = x + self.dwconv(x)
        x = x + self.mlp(self.norm2(x))

        return x


class ChannelBlock(nn.Module):
    def __init__(self, dim, mlp_ratio = 2.0):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.se = SEBlock(dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm2 = nn.BatchNorm2d(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False)
        )

    def forward(self, x):
        x = x + self.se(self.norm1(x))
        x = x + self.dwconv(x)
        x = x + self.mlp(self.norm2(x))
        return x



class TinyNeXt(nn.Module):
    def __init__(
        self,
        depths  = [3, 3, 6, 2],
        widths = [32, 64, 96, 192],
        exp_ratios = [2.0, 2.0, 2.0, 2.0],
        mlp_ratio = 2.0
    ):
        super().__init__()
        self.stem = Stem(widths[0])
        
        stages = []
        in_channels = widths[0]
        
        for i in range(4):
            if i > 0:
                stages.append(Downsample(in_channels, widths[i]))
                in_channels = widths[i]
                
            for _ in range(depths[i]):
                if i < 2:  
                    stages.append(ConvBlock(in_channels, widths[i], exp_ratios[i]))
                elif i == 2:  
                    stages.append(SpatialBlock(in_channels, mlp_ratio))
                else: 
                    stages.append(ChannelBlock(in_channels, mlp_ratio))
                    
        self.stages = nn.Sequential(*stages)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        features = self.stages(x) # (B, C, H, W)
        embedding = nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, C)
        return embedding


def TinyNeXt_T():
    return TinyNeXt(
        depths=[3, 3, 6, 2], 
        widths=[32, 64, 96, 192], 
        exp_ratios=[2.0]*4
    )

def TinyNeXt_S():
    return TinyNeXt(
        depths=[3, 3, 8, 3], 
        widths=[32, 64, 96, 192], 
        exp_ratios=[2.0]*4
    )

def TinyNeXt_M():
    return TinyNeXt(
        depths=[4, 4, 9, 4], 
        widths=[32, 64, 128, 256], 
        exp_ratios=[2.0, 2.0, 2.0, 1.5]
    )