import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import BrightnessDataset
from model import get_model

def train_model(data_dir='dataset', epochs=10, batch_size=16, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    full_dataset = BrightnessDataset(root_dir=data_dir, transform=transform)
    
    if len(full_dataset) < 2:
        print(f"Only {len(full_dataset)} images found in the dataset directory. Need at least 2 for training/val split.")
        print("Please add images to dataset/dark and dataset/very_dark.")
        return
        
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = get_model(num_classes=2).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / max(total, 1)
        
        # Validation
        if len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
            val_acc = 100. * correct / max(total, 1)
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {running_loss/max(len(train_loader), 1):.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss/max(len(val_loader), 1):.4f} | Val Acc: {val_acc:.2f}%')
            
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'model.pth')
                print('Saved best model!')
        else:
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {running_loss/max(len(train_loader), 1):.4f} | Train Acc: {train_acc:.2f}%')
            torch.save(model.state_dict(), 'model.pth')
            print('Saved model.pth (no validation set)')

if __name__ == '__main__':
    train_model()
