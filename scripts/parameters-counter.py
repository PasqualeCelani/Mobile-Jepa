import torch


import sys
import os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from models.ViT import * 
from models.Mobile_JEPA import MobileJEPA_Encoder, MobileJEPA_Predictor, TransformerPredictor 
from utils.config import get_config



def main():
    ############################## params ################################
    params = get_config("../training_results/round10/params.json")

    img_size = params["model_params"]["img_size"][0]
    patch_size = params["model_params"]["patch_size"]    
    features = params["model_params"]["features"]      

    embed_dim = params["model_params"]["embed_dim"]
    num_heads =  params["model_params"]["num_heads"]
    predictor_embed_dim = params["model_params"]["predictor_embed_dim"]
    dropout = params["model_params"]["dropout"]
    IS_VIT_BASED = True

    ######################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not IS_VIT_BASED:
        encoder = MobileJEPA_Encoder(img_size, features=features, is_target=False)
        target_encoder = MobileJEPA_Encoder(img_size, features=features, is_target=True)
        
        target_encoder.load_state_dict(encoder.state_dict())

        predictor = TransformerPredictor(img_size, patch_size, 256, 128)
    else:
        encoder = ViT_TinyL(img_size, patch_size, embed_dim, num_heads, dropout)
        target_encoder = ViT_TinyL(img_size, patch_size, embed_dim, num_heads, dropout)

        target_encoder.load_state_dict(encoder.state_dict())

        predictor = ViT_Predictor(img_size, patch_size, embed_dim, num_heads, dropout, predictor_embed_dim) 
    
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total encoder trainable parameters: {total_params:,}")

    total_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Total encoder trainable parameters: {total_params:,}")



if __name__ == "__main__":
    main()