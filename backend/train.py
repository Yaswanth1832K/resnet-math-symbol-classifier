import os
import random
import time
from shutil import copy2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub

def download_dataset():
    print("Downloading dataset from Kaggle...")
    # This downloads to a cache directory
    path = kagglehub.dataset_download("sagyamthapa/handwritten-math-symbols")
    print(f"Dataset downloaded to: {path}")
    
    # The dataset has an 'extracted_images' or similar folder. Let's find the inner 'dataset' folder.
    data_dir = path
    if os.path.exists(os.path.join(path, "dataset")):
        data_dir = os.path.join(path, "dataset")
    elif os.path.exists(os.path.join(path, "extracted_images")):
       data_dir = os.path.join(path, "extracted_images")
       
    return data_dir

def setup_data(data_dir, output_dir="data_split"):
    classes = [cls for cls in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, cls))]
    print("Classes found:", classes)
    
    random.seed(42)
    train_root = os.path.join(output_dir, "train")
    val_root = os.path.join(output_dir, "val")
    test_root = os.path.join(output_dir, "test")

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)
    
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    for cls in classes:
        src_cls = os.path.join(data_dir, cls)
        if not os.path.isdir(src_cls): continue
        
        images = os.listdir(src_cls)
        random.shuffle(images)
        n = len(images)
        
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img in train_imgs:
            os.makedirs(os.path.join(train_root, cls), exist_ok=True)
            copy2(os.path.join(src_cls, img), os.path.join(train_root, cls, img))
        for img in val_imgs:
            os.makedirs(os.path.join(val_root, cls), exist_ok=True)
            copy2(os.path.join(src_cls, img), os.path.join(val_root, cls, img))
        for img in test_imgs:
            os.makedirs(os.path.join(test_root, cls), exist_ok=True)
            copy2(os.path.join(src_cls, img), os.path.join(test_root, cls, img))
            
    return train_root, val_root, test_root, classes

def train_model(train_root, val_root, test_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_root, transform=train_transforms)
    val_data = datasets.ImageFolder(val_root, transform=val_transforms)
    test_data = datasets.ImageFolder(test_root, transform=val_transforms)

    num_workers = 0 if os.name == 'nt' else 2
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    num_classes = len(train_data.classes)
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 10
    
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {(time.time()-start):.1f}s")
        scheduler.step()

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet18_math_symbols.pth")
    print("Saved model weights to models/resnet18_math_symbols.pth")
    
    # Save classes map
    import json
    with open("models/classes.json", "w") as f:
        json.dump(train_data.classes, f)

if __name__ == "__main__":
    data_dir = download_dataset()
    train_root, val_root, test_root, classes = setup_data(data_dir)
    train_model(train_root, val_root, test_root)
