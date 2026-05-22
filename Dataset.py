from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms
from BlockMasking import *


class Imagenet100(Dataset):
    def __init__(self, hf_split, transform, is_labeled=False):
        self.hf_split = hf_split
        self.transform = transform
        self.is_labeled=is_labeled
        
    def __len__(self):
        return len(self.hf_split)
        
    def __getitem__(self, idx):
        item = self.hf_split[idx]
        img = item["image"].convert("RGB")
        t_img = self.transform(img)

        if not self.is_labeled: return t_img

        return t_img, item["label"] 

    

def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
):
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=crop_scale),
        transforms.ToTensor(),
        transforms.Normalize(normalization[0], normalization[1])
    ])


def get_dataloader(batch_size=64, img_size=224, mask_params=None):
    # Unpack mask parameters
    patch_size = mask_params["patch_size"]
    pred_mask_scale = mask_params["pred_mask_scale"]
    enc_mask_scale = mask_params["enc_mask_scale"]
    aspect_ratio = mask_params["aspect_ratio"]
    num_enc_masks = mask_params["num_enc_masks"]
    num_pred_masks = mask_params["num_pred_masks"]
    allow_overlap = mask_params["allow_overlap"]
    min_keep = mask_params["min_keep"]

    dataset = load_dataset("clane9/imagenet-100")
    transform = make_transforms(crop_size=img_size)
    

    train_ds = Imagenet100(dataset["train"], transform, False)


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


    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        collate_fn=mask_collator, 
        num_workers=6, 
        pin_memory=True, 
        persistent_workers=True,  
        drop_last=True
    )
    
    return train_loader


def get_linear_probe_dataloaders(batch_size=256, img_size=224):
    dataset = load_dataset("clane9/imagenet-100")
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_ds = Imagenet100(dataset["train"], train_transform, True)
    val_ds = Imagenet100(dataset["validation"], val_transform, True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader