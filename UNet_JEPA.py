import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UNet_Encoder(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, features)

        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        self.down4 = Down(features * 8, features * 8) 

        self.norm = nn.GroupNorm(num_groups=1, num_channels=features * 8)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) 

        return self.norm(x5)


class UNetJEPA_Encoder(nn.Module):
    def __init__(self, img_size=224, features=64, is_target=False):
        super().__init__()

        self.is_target = is_target
        self.cnn = UNet_Encoder(in_channels=3, features=features)

    def forward(self, x, masks=None):
        if self.is_target:
            return self.cnn(x) # [B, 512, 14, 14]
        
        assert masks is not None, "Context Encoder requires bounding box masks"

        crops = self._extract_crops(x, masks)

        return self.cnn(crops) # [B * nenc, 512, H/16, W/16]

    def _extract_crops(self, imgs, masks):
        crops = []
        B, nenc, _ = masks.shape

        for b in range(B):
            for i in range(nenc):
                y, x, h, w = masks[b, i].long().tolist()
                h = min(h, imgs.shape[2] - y)
                w = min(w, imgs.shape[3] - x)
                crop = imgs[b, :, y:y+h, x:x+w]
                crops.append(crop)

        return torch.stack(crops, dim=0)



class PredictorNet(nn.Module):
    def __init__(self, in_channels=512, features=512):
        super().__init__()

        self.conv1 = DoubleConv(in_channels, features)
        self.conv2 = DoubleConv(features, features)
        self.conv3 = DoubleConv(features, features)

        self.outc = nn.Conv2d(features, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return self.outc(x)



class UNetJEPA_Predictor(nn.Module):
    def __init__(self, img_size=224, features=64):
        super().__init__()
        self.img_size = img_size
        self.bottleneck_size = img_size // 16  # 14
        self.embed_dim = features * 8  # 512 channels
        
        # Learnable token for the 14x14 void
        self.mask_token = nn.Parameter(torch.randn(1, self.embed_dim, 1, 1))

        nn.init.normal_(self.mask_token, std=0.02)
        
        self.cnn = PredictorNet(in_channels=self.embed_dim, features=self.embed_dim)

    def forward(self, context_feats, target_feats, masks_enc, masks_pred):
        B = target_feats.shape[0]
        nenc = masks_enc.shape[1]
        npred = masks_pred.shape[1]
        
        # Create the 14x14 learnable canvas
        canvas = self.mask_token.expand(B, -1, self.bottleneck_size, self.bottleneck_size).clone()
        
        # Paste context features (Scaled down by 16) 
        context_feats = context_feats.view(B, nenc, self.embed_dim, context_feats.shape[2], context_feats.shape[3])
        
        for b in range(B):
            for i in range(nenc):
                # Divide pixel coordinates by 16 to get bottleneck coordinates
                y, x, h, w = (masks_enc[b, i] // 16).long().tolist()
                h = min(h, self.bottleneck_size - y)
                w = min(w, self.bottleneck_size - x)

                if h > 0 and w > 0:
                    canvas[b, :, y:y+h, x:x+w] = context_feats[b, i, :, :h, :w]
                

        predicted_full_map = self.cnn(canvas) # [B, 512, 14, 14]
        
        # Extract prediction blocks 
        pred_blocks = []
        target_blocks = []
        
        for b in range(B):
            for i in range(npred):
                # Divide pixel coordinates by 16
                y, x, h, w = (masks_pred[b, i] // 16).long().tolist()
                h = min(h, self.bottleneck_size - y)
                w = min(w, self.bottleneck_size - x)
                
                if h > 0 and w > 0:
                    pred_blocks.append(predicted_full_map[b, :, y:y+h, x:x+w])
                    target_blocks.append(target_feats[b, :, y:y+h, x:x+w])
                
        pred_blocks = torch.stack(pred_blocks, dim=0)
        target_blocks = torch.stack(target_blocks, dim=0)
        
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