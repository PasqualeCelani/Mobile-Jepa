import torch
from ViT import *

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    encoder = ViT_TinyL()
    predictor = ViT_Predictor()

    encoder.to(device)
    predictor.to(device)


if __name__ == "__main__":
    main()