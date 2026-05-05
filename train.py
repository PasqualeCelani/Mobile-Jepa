import torch
from ViT import *
from BlockMasking import *

def main():
    ############################## params ################################
    img_size = (224, 224)
    patch_size=16
    embed_dim=192
    num_heads = 3
    predictor_embed_dim=92
    dropout=0.1

    #Masking params
    enc_mask_scale=(0.2, 0.8)
    pred_mask_scale=(0.2, 0.8)
    aspect_ratio=(0.3, 3.0)
    num_enc_masks=1
    num_pred_masks=2
    min_keep=4
    allow_overlap=False
    ######################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    encoder = ViT_TinyL(
        img_size[0], patch_size, embed_dim, num_heads, dropout
    )

    target_encoder = ViT_TinyL(
        img_size[0], patch_size, embed_dim, num_heads, dropout
    )

    predictor = ViT_TinyL(
        img_size[0], patch_size, embed_dim, num_heads, dropout, predictor_embed_dim
    )

    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)

    mask_collator = MaskCollator(
        input_size=img_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep
    )

if __name__ == "__main__":
    main()