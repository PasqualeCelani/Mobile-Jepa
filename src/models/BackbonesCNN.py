import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.GELU()
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


class CNN_Encoder(nn.Module):
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

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet18_Encoder(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )


        self.layer1 = self._make_layer(features, features, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(features, features * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(features * 2, features * 4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(features * 4, features * 8, num_blocks=2, stride=2)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=features * 8)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResnetBlock(in_channels, out_channels, stride=s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.inc(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return self.norm(x)

class CIFAR_ResNet_Encoder(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )


        self.layer1 = self._make_layer(features, features, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(features, features * 2, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(features * 2, features * 4, num_blocks=3, stride=2)
        
        self.norm = nn.GroupNorm(num_groups=1, num_channels=features * 4)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResnetBlock(in_channels, out_channels, stride=s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.inc(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return self.norm(x)

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
