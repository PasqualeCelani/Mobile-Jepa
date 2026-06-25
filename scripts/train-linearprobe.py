import torch
import torch.nn as nn

import sys
import os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))


from models.Mobile_JEPA import MobileJEPA_Encoder, LinearProbeJEPA
from data.Dataset import get_linear_probe_dataloaders
from utils.config import get_config

def main():
    params = get_config("../training_results/params.json")

    img_size = params["model_params"]["img_size"][0]   
    features = params["model_params"]["features"] 
    batch_size = params["test_params"]["linear"]["batch_size"]
    lr = params["test_params"]["linear"]["lr"]
    momentum = params["test_params"]["linear"]["momentum"]
    weight_decay = params["test_params"]["linear"]["weight_decay"]
    T_max = params["test_params"]["linear"]["T_max"]
    num_classes = params["test_params"]["linear"]["num_classes"]
    dataset_name = params["test_params"]["knn"]["dataset-name"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = MobileJEPA_Encoder(img_size=img_size, features=features, is_target=True)
    
    checkpoint = torch.load("checkpoint.pth", map_location=device)
    encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
    print(f"Loaded JEPA Target Encoder weights from epoch {checkpoint['epoch']}")

    model = LinearProbeJEPA(encoder, embed_dim=features * 8, num_classes=num_classes).to(device)

    train_loader, val_loader = get_linear_probe_dataloaders(batch_size=batch_size, img_size=img_size, dataset_name=dataset_name)

    optimizer = torch.optim.SGD(model.head.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    criterion = nn.CrossEntropyLoss()
    
    epochs = params["test_params"]["linear"]["epochs"]
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        train_acc = 100. * correct / total
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_linear_probe.pth")
            print(f"Saved new best model with Val Acc: {best_val_acc:.2f}%")

    print(f"\n Best Validation Accuracy: {best_val_acc:.2f}%")
    

if __name__ == "__main__":
    main()