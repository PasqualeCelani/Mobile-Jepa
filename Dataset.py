from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms


class Imagenet100(Dataset):
    def __init__(self, hf_split, transform):
        self.hf_split = hf_split
        self.transform = transform
        
    def __len__(self):
        return len(self.hf_split)
        
    def __getitem__(self, idx):
        img = self.hf_split[idx]["image"].convert("RGB")
        return self.transform(img)  


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


def get_dataloader(batch_size=64, img_size=224):
    dataset = load_dataset("clane9/imagenet-100")
    transform = make_transforms(crop_size=img_size)
    

    train_ds = Imagenet100(dataset["train"], transform)


    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        num_workers=6, 
        pin_memory=True, 
        persistent_workers=True,  
        drop_last=True
    )
    
    return train_loader