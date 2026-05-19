import torch
from ViT import *
from TinyNeXt import *
from Dataset import get_dataloader
from Schedulers import *
from PatchExtractor import PatchCNNExtractor
import torch.nn.functional as F

def main():
    ############################## params ################################
    #ViTs params
    img_size = 224
    patch_size = 56  
    num_patches = (img_size // patch_size) ** 2 
    embed_dim = 192           # Target embedding dim for predictor
    predictor_embed_dim = 96  # Internal predictor dimension
    num_heads = 3
    dropout = 0.1

    
    #Training parameters
    batch_size = 64
    warmup = 40
    start_lr = 0.0002
    lr = 0.001
    final_lr = 1.0e-06
    final_weight_decay = 0.4
    ipe_scale = 1.0
    epochs = 1
    weight_decay = 0.04
    mask_ratio = 0.3  
    ######################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cnn_backbone = TinyNeXt_M()
    
    extractor = PatchCNNExtractor(
        cnn_model=cnn_backbone,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        dropout=dropout
    )


    predictor = ViT_Predictor(
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        depth=12
    )

    extractor.to(device)
    predictor.to(device)


    data_loader = get_dataloader(batch_size, img_size)
    iterations_per_epoch = len(data_loader)

    param_groups = [
        {
            'params': (p for n, p in extractor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1)),
            'lr': lr,
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1)),
            'lr': lr,
        }, {
            'params': (p for n, p in extractor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0,
            'lr': lr,
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0,
            'lr': lr,
        }
    ]

    optimizer = torch.optim.AdamW(param_groups)

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*epochs*iterations_per_epoch))

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=weight_decay,
        final_wd=final_weight_decay,
        T_max=int(ipe_scale*epochs*iterations_per_epoch))
    


    start_epoch = 0 
    for epoch in range(start_epoch, epochs):
        print(f"Epoch: {epoch+1}")

        for udata in data_loader:
            imgs = udata.to(device, non_blocking=True)

            scheduler.step()
            wd_scheduler.step()

            # Extract patch embeddings + keep raw versions for loss
            embeddings_with_pos, raw_embeddings = extractor(
                imgs, return_unmasked=True
            )  # [B, 16, 192]
            
            # Generate random mask: True=keep (70%), False=mask (30%)
            # Shape: [B, num_patches]
            keep_prob = 1.0 - mask_ratio
            mask = torch.bernoulli(
                torch.full((num_patches,), keep_prob, device=device)
            ).bool()
            mask = mask.unsqueeze(0).expand(imgs.shape[0], -1)  # [B, N]
            
            # Forward through predictor WITH masking
            # Masked positions get replaced with learnable mask_token internally
            predictions = predictor(
                embeddings_with_pos,  # [B, N, embed_dim]
                mask=mask             # [B, N] boolean: True=keep
            )  # [B, N, embed_dim]
            
            # Compute L2 loss ONLY on masked positions
            # mask=False means this position was masked → compute loss here
            masked_positions = ~mask  # [B, N]: True where I masked
            num_masked = masked_positions.sum().item()
            
            if num_masked > 0:
                # Gather predictions and targets for masked positions
                pred_masked = predictions[masked_positions]      # [M, embed_dim]
                target_masked = raw_embeddings[masked_positions] # [M, embed_dim]
                
                # L2 loss (mean over masked patches and embedding dims)
                loss = F.mse_loss(pred_masked, target_masked, reduction='mean')
            else:
                # Edge case: no positions masked
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Loss: {loss.item():.4f} | Masked: {num_masked}/{num_patches}")


if __name__ == "__main__":
    main() 