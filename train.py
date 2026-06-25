import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from Mobile_JEPA import MobileJEPA_Encoder, MobileJEPA_Predictor 
from Dataset import get_dataloader
from Schedulers import WarmupCosineSchedule, CosineWDSchedule
from config import get_config
import matplotlib.pyplot as plt

def plot_loss_curve(epoch_list, loss_list, save_path="loss_ssl.png"):
    if not epoch_list:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, loss_list, marker='o', linestyle='-', markersize=3, label='Loss')
    plt.title('Training Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_checkpoint(epoch, encoder, target_encoder, predictor, optimizer, scheduler, wd_scheduler, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'target_encoder_state_dict': target_encoder.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'wd_scheduler_state_dict': wd_scheduler.state_dict(), 
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(filename, encoder, target_encoder, predictor, optimizer, scheduler, wd_scheduler):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        target_encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
        predictor.load_state_dict(checkpoint['predictor_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        wd_scheduler.load_state_dict(checkpoint['wd_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint '{filename}' (resuming from epoch {start_epoch})")
        return start_epoch
    return 0

def main():
    ############################## params ################################
    params = get_config("./params.json")

    img_size = params["model_params"]["img_size"][0]
    patch_size = params["model_params"]["patch_size"]    
    features = params["model_params"]["features"]      

    # Masking params
    mask_params = params["mask_params"]

    # Training parameters
    batch_size = params["training_params"]["batch_size"]     
    warmup = params["training_params"]["warmup"]
    start_lr = params["training_params"]["start_lr"]
    lr = params["training_params"]["lr"]
    final_lr = params["training_params"]["final_lr"]
    final_weight_decay = params["training_params"]["final_weight_decay"]
    ipe_scale = params["training_params"]["ipe_scale"]
    epochs = params["training_params"]["epochs"]
    weight_decay =  params["training_params"]["weight_decay"]
    ema = params["training_params"]["ema"]
    dataset_name = params["training_params"]["dataset-name"]
    ######################################################################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    encoder = MobileJEPA_Encoder(img_size, features=features, is_target=False)
    target_encoder = MobileJEPA_Encoder(img_size, features=features, is_target=True)
    
    target_encoder.load_state_dict(encoder.state_dict())
    
    predictor = MobileJEPA_Predictor(img_size, features=features) 

    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)

    data_loader = get_dataloader(batch_size, img_size, mask_params, dataset_name)
    iterations_per_epoch = len(data_loader)

    param_groups = [
        {'params': (p for n, p in encoder.named_parameters() if ('bias' not in n) and (len(p.shape) != 1))}, 
        {'params': (p for n, p in predictor.named_parameters() if ('bias' not in n) and (len(p.shape) != 1))}, 
        {'params': (p for n, p in encoder.named_parameters() if ('bias' in n) or (len(p.shape) == 1)), 'WD_exclude': True, 'weight_decay': 0}, 
        {'params': (p for n, p in predictor.named_parameters() if ('bias' in n) or (len(p.shape) == 1)), 'WD_exclude': True, 'weight_decay': 0}
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay, lr=lr)

    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr, ref_lr=lr, final_lr=final_lr,
        T_max=int(ipe_scale * epochs * iterations_per_epoch))

    wd_scheduler = CosineWDSchedule(
        optimizer, ref_wd=weight_decay, final_wd=final_weight_decay,
        T_max=int(ipe_scale * epochs * iterations_per_epoch))
    

    for p in target_encoder.parameters():
        p.requires_grad = False
    
    total_steps = int(iterations_per_epoch * epochs * ipe_scale)
    checkpoint_path = "checkpoint.pth"
    start_epoch = load_checkpoint(checkpoint_path, encoder, target_encoder, predictor, optimizer, scheduler, wd_scheduler)

    epoch_numbers = []
    epoch_losses  = []

    try:
        for epoch in range(start_epoch, epochs):
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch")
            
            for i, (imgs, masks_enc, masks_pred) in enumerate(progress_bar):
                imgs = imgs.to(device, non_blocking=True)
                masks_enc = masks_enc.to(device, non_blocking=True)
                masks_pred = masks_pred.to(device, non_blocking=True)

                scheduler.step()
                wd_scheduler.step()


                with torch.no_grad():
                    target_feats = target_encoder(imgs) 
                    # Shape: [B, 64, 224, 224]


                context_feats = encoder(imgs, masks_enc) 
                # Shape: [B * nenc, 64, H_crop, W_crop]


                # The predictor builds the canvas, runs the U-Net, and extracts the specific blocks
                pred_blocks, target_blocks = predictor(context_feats, target_feats, masks_enc, masks_pred)
                # Shapes: [B * npred, 64, H_pred, W_pred]


                loss = F.smooth_l1_loss(pred_blocks, target_blocks)
                

                sim = F.cosine_similarity(pred_blocks, target_blocks, dim=1).mean().item()
                
                if i % 100 == 0:
                    progress_bar.set_description(
                        f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Sim: {sim:.4f}"
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                global_step = epoch * iterations_per_epoch + i
                m = ema[0] + global_step * (ema[1] - ema[0]) / total_steps

                with torch.no_grad():
                    for name, param_k in target_encoder.named_parameters():
                        param_q = encoder.get_parameter(name)
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)
            
            epoch_numbers.append(epoch + 1)
            epoch_losses.append(loss.item())

            save_checkpoint(epoch, encoder, target_encoder, predictor, optimizer, scheduler, wd_scheduler, checkpoint_path)
    except KeyboardInterrupt:
        plot_loss_curve(epoch_numbers, epoch_losses)

    plot_loss_curve(epoch_numbers, epoch_losses)

if __name__ == "__main__":
    main()