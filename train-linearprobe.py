import torch
from UNet_JEPA import UNetJEPA_Encoder, LinearProbeJEPA
from Dataset import get_linear_probe_dataloaders
import torch.nn as nn

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    encoder = UNetJEPA_Encoder(img_size=224, patch_size=16, embed_dim=192, is_target=True)
    
    checkpoint = torch.load("checkpoint.pth", map_location=device)
    

    encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
    print(f"Loaded JEPA Target Encoder weights from epoch {checkpoint['epoch']}")
    

    model = LinearProbeJEPA(encoder, embed_dim=192, num_classes=100).to(device)
    

    train_loader, val_loader = get_linear_probe_dataloaders(batch_size=256)
    

    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)
    criterion = nn.CrossEntropyLoss()
    

    epochs = 90
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

if __name__ == "__main__":
    main()