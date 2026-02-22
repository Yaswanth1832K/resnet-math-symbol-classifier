#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# In[2]:


import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")


# In[3]:


import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from shutil import copy2
import time


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
if device.type == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))


# In[5]:


data_dir = "C:/Users/yaswa/Downloads/archive/dataset"  # adjust to your dataset path

classes = [cls for cls in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, cls))]
print("Classes found:", classes)
print("Number of classes:", len(classes))


# In[6]:


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # convert grayscale → 3 channels
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = val_transforms


# In[7]:


random.seed(42)

src = data_dir
dst_root = "data_split"
train_root = os.path.join(dst_root, "train")
val_root = os.path.join(dst_root, "val")
test_root = os.path.join(dst_root, "test")

os.makedirs(train_root, exist_ok=True)
os.makedirs(val_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

# Ratios
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

for cls in classes:
    src_cls = os.path.join(src, cls)
    if not os.path.isdir(src_cls):
        continue
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

print("Data split complete!")


# In[8]:


train_data = datasets.ImageFolder(train_root, transform=train_transforms)
val_data = datasets.ImageFolder(val_root, transform=val_transforms)
test_data = datasets.ImageFolder(test_root, transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

print("Train size:", len(train_data))
print("Val size:", len(val_data))
print("Test size:", len(test_data))
print("Classes:", train_data.classes)


# In[9]:


model = models.resnet18(pretrained=True)

# Freeze earlier layers
for param in model.parameters():
    param.requires_grad = False

# Replace final fc layer
num_features = model.fc.in_features
num_classes = len(train_data.classes)
model.fc = nn.Linear(num_features, num_classes)

model = model.to(device)
print(model)


# In[10]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# In[ ]:


torch.backends.cudnn.benchmark = True

num_epochs = 15
train_losses, val_losses, val_accuracies = [], [], []

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
    train_losses.append(epoch_loss)
    
    # validation
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
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {epoch_loss:.4f} | "
          f"Val Loss: {val_epoch_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}% | "
          f"Time: {(time.time()-start):.1f}s")
    
    scheduler.step()


# In[ ]:


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curves")

plt.subplot(1,2,2)
plt.plot(val_accuracies, label="Val Accuracy")
plt.legend()
plt.title("Validation Accuracy (%)")

plt.show()


# In[15]:


model.eval()
correct, total = 0, 0
all_labels, all_preds = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=train_data.classes))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))


# In[16]:


torch.save(model.state_dict(), "resnet_math_symbols_weights.pth")
print("Saved model weights.")

torch.save(model, "resnet_math_symbols_fullmodel.pth")
print("Saved full model.")


# In[ ]:




