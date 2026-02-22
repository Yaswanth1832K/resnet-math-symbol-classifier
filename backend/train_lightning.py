import os
import random
import time
from shutil import copy2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import json

def setup_data(data_dir, output_dir="data_split_lightning"):
    classes = sorted([cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))])
    print(f"Classes found ({len(classes)}): {classes[:10]}...")
    
    random.seed(42)
    train_root = os.path.join(output_dir, "train")
    val_root = os.path.join(output_dir, "val")

    if os.path.exists(output_dir):
        print("Data split already exists. Skipping setup.")
        return train_root, val_root, classes

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)
    
    train_ratio = 0.85
    max_images_per_class = 50 # Tiny subset for fast iteration

    for cls in classes:
        src_cls = os.path.join(data_dir, cls)
        images = os.listdir(src_cls)
        random.shuffle(images)
        
        images = images[:max_images_per_class]

        n = len(images)
        n_train = int(train_ratio * n)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:]

        os.makedirs(os.path.join(train_root, cls), exist_ok=True)
        os.makedirs(os.path.join(val_root, cls), exist_ok=True)

        for img in train_imgs:
            copy2(os.path.join(src_cls, img), os.path.join(train_root, cls, img))
        for img in val_imgs:
            copy2(os.path.join(src_cls, img), os.path.join(val_root, cls, img))
            
    return train_root, val_root, classes

def train_model(train_root, val_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_root, transform=train_transforms)
    val_data = datasets.ImageFolder(val_root, transform=val_transforms)

    num_workers = 0 # FORCED NO MULTIPROCESSING
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(train_data.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    num_epochs = 1 # fast check
    
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
            if i % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {(time.time()-start):.1f}s")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet18_math_symbols_v3.pth")
    print("Saved model weights to models/resnet18_math_symbols_v3.pth")
    
    with open("models/classes_v3.json", "w") as f:
        json.dump(train_data.classes, f)

if __name__ == "__main__":
    data_dir = r"C:\Users\yaswa\.cache\kagglehub\datasets\xainano\handwrittenmathsymbols\versions\2\extracted_full\extracted_images"
    train_root, val_root, classes = setup_data(data_dir)
    train_model(train_root, val_root)
