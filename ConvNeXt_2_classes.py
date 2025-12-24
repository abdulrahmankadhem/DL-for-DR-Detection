import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, entropy
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

matplotlib.use('Agg') # 'Agg' backend is used so that plots can be saved without a display

# Dataset 

class APTOSBinaryDataset(Dataset):
    def __init__(self, dataframe, transform=None):

        self.data = dataframe.copy()
        self.transform = transform
        
        self.data['binary_label'] = self.data['diagnosis'].apply(lambda x: 0 if x == 0 else 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(row['img_dir'], row['id_code'] + ".png")  
        image = Image.open(img_path).convert("RGB")
        label = row['binary_label']
        if self.transform:
            image = self.transform(image)
        return image, label

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
       self.clip_limit = clip_limit
       self.tile_grid_size = tile_grid_size

    def __call__(self, img):
      

      clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
      img_np = np.array(img)
      r, g, b = cv2.split(img_np)
      g_clahe = clahe.apply(g)

      
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
      tophat = cv2.morphologyEx(g_clahe, cv2.MORPH_TOPHAT, kernel)
      enhanced = cv2.add(g_clahe, tophat)
      merged = cv2.merge([r, enhanced, b])
      return Image.fromarray(merged)


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Parameters

if __name__ == "__main__":

    train_img_dir = "" # Paths used for images and csvs from the APTOS dataset
    train_csv = ""

    val_img_dir = ""
    val_csv = ""

    test_img_dir = ""
    test_csv = ""

    save_path = "" # Path to save the best model

    num_classes = 2
    batch_size = 64 # Batch size can be adjusted based on GPU memory
    num_epochs = 50
    total_start = time.time()

    # Data Preprocessing
    
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    df_train["img_dir"] = train_img_dir
    df_val["img_dir"] = val_img_dir
    df_test["img_dir"] = test_img_dir

    combined_df = pd.concat([df_test, df_val, df_train], ignore_index=True)

    train_df, test_df = train_test_split(
    combined_df,
    test_size=0.2,
    stratify=combined_df["diagnosis"],
    random_state=42)


    def get_preprocessing_transform(img_size=224):
     return transforms.Compose([
        CLAHETransform(),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.1), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = get_preprocessing_transform(img_size=224)
    
    def get_test_transform(img_size=224):
     return transforms.Compose([
        CLAHETransform(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = get_test_transform(img_size=224)
    test_dataset = APTOSBinaryDataset(test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Num workers can be adjusted based on system

    train_dataset = APTOSBinaryDataset(train_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Converts dataset labels to binary

    train_df["diagnosis"] = train_df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)
    test_df["diagnosis"] = test_df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)

    model = models.convnext_tiny(weights="IMAGENET1K_V1")
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1) # Adjusts model head for binary classification
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()  
    optimizer = torch.optim.AdamW([
    {'params': model.features.parameters(), 'lr': 3e-5},
    {'params': model.classifier.parameters(), 'lr': 3e-4}
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    # Training Loop

    early_stopping = EarlyStopping(patience=9)

    best_acc = 0.0
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accs = []

    for epoch in range(num_epochs):
        epoch_start = time.time()

        model.train()
        running_train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)  # float labels for BCE
            optimizer.zero_grad()

            outputs = model(imgs).squeeze()  # shape: [B]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_losses.append(epoch_train_loss)

        # Validation

        model.eval()
        running_val_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.float().to(device)
                outputs = model(imgs).squeeze()

                val_loss = criterion(outputs, labels).item()
                running_val_loss += val_loss

                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).long()

                preds.extend(predicted.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        epoch_val_loss = running_val_loss / len(test_loader)
        epoch_val_losses.append(epoch_val_loss)

        acc = accuracy_score(targets, preds)
        epoch_val_accs.append(acc)

        epoch_end = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {acc:.4f} | Time: {epoch_end-epoch_start:.2f}s")

        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print("Saved Best Model!")

        early_stopping(acc)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    
    # Evaluation

    total_end = time.time()
    total_time = total_end - total_start
    epoch_reached = epoch + 1
    avg_epoch_time = total_time / epoch_reached

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # For unique file naming

    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(base_dir, "plots_convtbi") # Directory to save plots
    os.makedirs(plot_dir, exist_ok=True)

    # Load best model
    state_dict = torch.load(save_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            # Binary: outputs shape = [batch, 1]
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy())

            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:\n", cm)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_convtbi_{timestamp}.png"))
    print(f"Saved confusion_matrix_convtbi_{timestamp}.png")
    plt.close()
  
  # Sensitivity & Specificity

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)      # Recall of positive class
    specificity = tn / (tn + fp)      # Recall of negative class

    print(f"Sensitivity (Recall+): {sensitivity:.4f}")
    print(f"Specificity (Recall-): {specificity:.4f}")

    # Accuracy
    acc = accuracy_score(all_targets, all_preds)
    print(f"Accuracy: {acc:.5f}")

    # Precision
    Binary_precision = precision_score(all_targets, all_preds, average='binary')
    print(f"Binary Precision: {Binary_precision:.4f}")

    # F1 Score
    Binary_f1 = f1_score(all_targets, all_preds, average='binary')
    print(f"Binary F1 Score: {Binary_f1:.5f}")

    # Kurtosis & Entropy (on predicted probabilities)
    kurt = kurtosis(all_probs, axis=None)
    eps = 1e-7
    ent = -np.mean(all_probs * np.log(all_probs + eps) + (1 - all_probs) * np.log(1 - all_probs + eps))
    print(f"Kurtosis: {kurt:.5f}")
    print(f"Entropy: {ent:.5f}")

    print(f"Total Training Time: {total_time/60:.2f} minutes")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f} seconds")

    # ROC Curve
    plt.figure()

    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')  # Reference line

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Positive Class Only')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"roc_curve_convtbi_{timestamp}.png"))
    print(f"Saved roc_curve_convtbi_{timestamp}.png")
    plt.close()
    
    # Probability Distribution Histogram
    plt.figure()
    plt.hist(all_probs[all_targets == 0], bins=30, alpha=0.6, label="Class 0")
    plt.hist(all_probs[all_targets == 1], bins=30, alpha=0.6, label="Class 1")
    plt.axvline(all_probs[all_targets == 0].mean(), linestyle='--')
    plt.axvline(all_probs[all_targets == 1].mean(), linestyle='--')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Probability Distribution by True Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"prob_hist_convtbi_{timestamp}.png"))
    plt.close()

    # Accuracy and Loss vs Epoch

    plt.figure()

    # Left y-axis for Accuracy
    ax1 = plt.gca()
    ax1.plot(range(1, len(epoch_val_accs)+1), epoch_val_accs, color='tab:blue', label='Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 1)  # Accuracy is 0-1

    # Right y-axis for Loss
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(epoch_val_losses)+1), epoch_val_losses, color='tab:red', label='Loss', linewidth=2)
    ax2.set_ylabel('Loss')
    ax2.tick_params(axis='y')

    plt.title('Accuracy and Loss vs Epoch')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"accuracy_loss_vs_epoch_convtbi_{timestamp}.png"))
    plt.close()
    print(f"Saved accuracy_loss_vs_epoch_convtbi_{timestamp}.png")