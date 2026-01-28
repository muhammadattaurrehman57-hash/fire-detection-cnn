# Fire vs Non-Fire Detection
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import MobileNet_V2_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dataset path
DATA_DIR= "/content/drive/MyDrive/fire_dataset/"
# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
# Load dataset
dataset= datasets.ImageFolder(DATA_DIR, transform=transform)
class_names =dataset.classes
print("Classes:", class_names)
print("Total images:", len(dataset))
# Train / Val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Load MobileNetV2
weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
# Freeze early layers, fine-tune last layers
for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-4:].parameters():
    param.requires_grad = True
# replace classifier
model.classifier[1] = nn.Linear(model.last_channel, 1)
model = model.to(device)
# loss & optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# training
EPOCHS = 10
train_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    # validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds = (probs > 0.4).long()

            correct += (preds.squeeze(1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    val_accuracies.append(acc)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} Val Acc: {acc:.4f}")

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long().squeeze(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Visualization
model.eval()

fire_imgs = []
nonfire_imgs = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)

        for i in range(len(images)):
            img = images[i]
            prob = probs[i].item()
            label = labels[i].item()

            if label == 0 and len(fire_imgs) < 3:
                fire_imgs.append((img, prob))
            elif label == 1 and len(nonfire_imgs) < 3:
                nonfire_imgs.append((img, prob))

            if len(fire_imgs) == 3 and len(nonfire_imgs) == 3:
                break
        if len(fire_imgs) == 3 and len(nonfire_imgs) == 3:
            break
samples = fire_imgs + nonfire_imgs
plt.figure(figsize=(12,6))

for idx, (img, prob) in enumerate(samples):
    plt.subplot(2,3,idx+1)

    img = img.cpu().permute(1,2,0)
    img = img * torch.tensor([0.229, 0.224, 0.225]) + \
          torch.tensor([0.485, 0.456, 0.406])
    img = img.clamp(0,1)

    if prob < 0.5:
        pred_class = class_names[0]  # fire
        confidence = (1 - prob) * 100
    else:
        pred_class = class_names[1]  # non-fire
        confidence = prob * 100

    plt.imshow(img)
    plt.title(f"{pred_class}\n{confidence:.1f}%")
    plt.axis("off")

plt.show()
