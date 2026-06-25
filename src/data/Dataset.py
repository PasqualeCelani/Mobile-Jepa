from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Tuple, Optional
from datasets import load_dataset
import torchvision.transforms as transforms

import sys
import os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))


from utils.BlockMasking import *


@dataclass
class DatasetConfig:
    hf_repo: str                     # HuggingFace dataset repository name
    img_key: str                     # Key for the image in the HF dataset dict
    train_split: str = "train"       # Name of the training split
    val_split: str = "test"          # Name of the validation/test split
    
    # Transform parameters
    train_crop_size: int = 224       
    train_resize_size: Optional[Tuple[int, int]] = None # If set, uses Resize instead of RandomCrop
    probe_img_size: int = 224        
    
    # Normalization stats (mean, std)
    normalization: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        (0.485, 0.456, 0.406), 
        (0.229, 0.224, 0.225)
    )

DATASET_REGISTRY = {
    "imagenet-100": DatasetConfig(
        hf_repo="clane9/imagenet-100",
        img_key="image",
    ),
    "cifar-10": DatasetConfig(
        hf_repo="cifar10",
        img_key="img",
        train_crop_size=96,
        train_resize_size=(96, 96), 
        probe_img_size=96,
        normalization=((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ),
}


class HFDataset(Dataset):
    def __init__(self, hf_split, transform, is_labeled=False, img_key="image"):
        self.hf_split = hf_split
        self.transform = transform
        self.is_labeled=is_labeled
        self.img_key = img_key
        
    def __len__(self):
        return len(self.hf_split)
        
    def __getitem__(self, idx):
        item = self.hf_split[idx]
        img = item[self.img_key].convert("RGB")
        t_img = self.transform(img)

        if not self.is_labeled: return t_img

        return t_img, item["label"] 


def make_ssl_transforms(config: DatasetConfig, crop_scale=(0.3, 1.0)):
    pipeline = []
    
    if config.train_resize_size:
        pipeline.append(transforms.Resize(config.train_resize_size, interpolation=transforms.InterpolationMode.BILINEAR))
    else:
        pipeline.append(transforms.RandomResizedCrop(config.train_crop_size, scale=crop_scale))
        
    pipeline.extend([
        transforms.ToTensor(),
        transforms.Normalize(*config.normalization)
    ])
    
    return transforms.Compose(pipeline)

def make_linear_probe_transforms(config: DatasetConfig):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.probe_img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*config.normalization)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(config.probe_img_size + 32),
        transforms.CenterCrop(config.probe_img_size), 
        transforms.ToTensor(),
        transforms.Normalize(*config.normalization)
    ])
    
    return train_transform, val_transform



def get_dataloader(batch_size=64, img_size=224, mask_params=None, dataset_name="imagenet-100"):
    # Unpack mask parameters
    patch_size = mask_params["patch_size"]
    pred_mask_scale = mask_params["pred_mask_scale"]
    enc_mask_scale = mask_params["enc_mask_scale"]
    aspect_ratio = mask_params["aspect_ratio"]
    num_enc_masks = mask_params["num_enc_masks"]
    num_pred_masks = mask_params["num_pred_masks"]
    allow_overlap = mask_params["allow_overlap"]
    min_keep = mask_params["min_keep"]

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASET_REGISTRY.keys())}")
        
    config = DATASET_REGISTRY[dataset_name]
    hf_dataset = load_dataset(config.hf_repo)
    transform = make_ssl_transforms(config)
    
    train_ds =  HFDataset(
        hf_split=hf_dataset[config.train_split], 
        transform=transform, 
        is_labeled=False, 
        img_key=config.img_key
    )

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


def get_linear_probe_dataloaders(batch_size=256, img_size=224, dataset_name="imagenet-100"):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASET_REGISTRY.keys())}")
        
    config = DATASET_REGISTRY[dataset_name]
    hf_dataset = load_dataset(config.hf_repo)
    
    train_transform, val_transform = make_linear_probe_transforms(config)
    
    train_ds = HFDataset(hf_dataset[config.train_split], train_transform, True, config.img_key)
    val_ds = HFDataset(hf_dataset[config.val_split], val_transform, True, config.img_key)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    
    return train_loader, val_loader