import torch
from UNet_JEPA import UNetJEPA_Encoder, UNetJEPA_Predictor
from Dataset import get_dataloader
from Schedulers import *
import torch.nn.functional as F
import os
from tqdm import tqdm


def save_checkpoint(epoch, encoder, target_encoder, predictor, optimizer, scheduler, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'target_encoder_state_dict': target_encoder.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(filename, encoder, target_encoder, predictor, optimizer, scheduler):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        target_encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
        predictor.load_state_dict(checkpoint['predictor_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint '{filename}' (resuming from epoch {start_epoch})")
        return start_epoch
    return 0


#Same as ViT (techical dept)
def apply_masks(x, masks):
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)
    
def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

def main():
    ############################## params ################################
    #U-net params
    img_size = (224, 224)
    patch_size=16
    embed_dim=192
    predictor_embed_dim=96

    #Masking params
    enc_mask_scale=(0.85, 1.0)
    pred_mask_scale=(0.15, 0.2)
    aspect_ratio=(0.75, 1.5)
    num_enc_masks=1
    num_pred_masks=4
    min_keep=10
    allow_overlap=False

    mask_params = {
        "patch_size" : patch_size,
        "enc_mask_scale" : enc_mask_scale,
        "pred_mask_scale" : pred_mask_scale,
        "aspect_ratio" :  aspect_ratio,
        "num_enc_masks" : num_enc_masks,
        "num_pred_masks" : num_pred_masks,
        "min_keep" : min_keep,
        "allow_overlap" : allow_overlap
    }

    #Training parameters
    batch_size = 64
    warmup = 40
    start_lr =  0.0002
    lr =  0.001
    final_lr =  1.0e-06
    final_weight_decay =  0.4
    ipe_scale =  1.0
    epochs =  100
    weight_decay = 0.04
    ema = [0.996, 1.0]
    ######################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    encoder = UNetJEPA_Encoder(
        img_size[0], patch_size, embed_dim, is_target=False
    )

    target_encoder = UNetJEPA_Encoder(
        img_size[0], patch_size, embed_dim, is_target=True
    )

    predictor = UNetJEPA_Predictor(
        img_size[0], patch_size, embed_dim, predictor_embed_dim
    )

    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)

    data_loader = get_dataloader(batch_size, img_size[0], mask_params)
    iterations_per_epoch = len(data_loader)

    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
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
    
    for p in target_encoder.parameters():
        p.requires_grad = False

    # momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(iterations_per_epoch*epochs*ipe_scale)
                          for i in range(int(iterations_per_epoch*epochs*ipe_scale)+1))
    
    checkpoint_path = "checkpoint.pth"
    start_epoch = load_checkpoint(checkpoint_path, encoder, target_encoder, predictor, optimizer, scheduler)

    for epoch in range(start_epoch, epochs):
        print(f"Epoch: {epoch+1}")

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for i, (udata, masks_enc, masks_pred) in enumerate(progress_bar):
            imgs = udata.to(device, non_blocking=True)

            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]

            scheduler.step()
            wd_scheduler.step()

            with torch.no_grad():
                h = target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                B = len(h)
                h = apply_masks(h, masks_pred)
                h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
            
            z = encoder(imgs, masks_enc)
            z = predictor(z, masks_enc, masks_pred)

            loss = F.smooth_l1_loss(z, h)
            sim = F.cosine_similarity(z, h, dim=-1).mean().item()
            
            if i % 100 == 0:
                progress_bar.set_description(
                    f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Sim: {sim:.4f} | Target Latent Std: {h.std(dim=0).mean().item():.4f}"
                )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # momentum update of target encoder
            with torch.no_grad():
                m = next(momentum_scheduler)
                for name, param_k in target_encoder.named_parameters():
                    param_q = encoder.get_parameter(name)
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
        

        save_checkpoint(epoch, encoder, target_encoder, predictor, optimizer, scheduler, checkpoint_path)

if __name__ == "__main__":
    main()