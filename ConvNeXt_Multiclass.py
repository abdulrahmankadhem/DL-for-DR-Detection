import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, accuracy_score, cohen_kappa_score, precision_score
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, entropy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

matplotlib.use('Agg')

# Dataset

class APTOSDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)  
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]  
        img_id = row["id_code"]
        label = row["diagnosis"]
        img_dir = row["img_dir"]

        img_path = os.path.join(img_dir, img_id + ".png")
        image = Image.open(img_path).convert("RGB")

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

class FocalLossSmooth(nn.Module):
    def __init__(self, gamma=2.0, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        num_classes = logits.size(-1)

        # Label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        focal_weight = (1 - probs) ** self.gamma
        loss = -true_dist * focal_weight * log_probs

        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Parameters

if __name__ == "__main__":

    train_img_dir = "" # These are the paths for images and csvs from the APTOS dataset
    train_csv = ""

    val_img_dir = ""
    val_csv = ""

    test_img_dir = ""
    test_csv = ""

    save_path = "" # Path to save the best model

    num_classes = 5
    batch_size = 64 # Adjust based on GPU memory
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

    test_dataset = APTOSDataset(test_df, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4) 
    
    # num_workers can be adjusted based on system

    train_dataset = APTOSDataset(train_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_counts = np.bincount(train_df["diagnosis"].values)
    class_weights = 1. / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = models.convnext_tiny(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
    nn.Flatten(),                   
    nn.Dropout(p=0.2),               
    nn.Linear(model.classifier[2].in_features, num_classes)) # Redefines head architecture for 5 classes
    model = model.to(device)
    

    criterion = FocalLossSmooth(gamma=1.0, smoothing=0.1, reduction='none') # Focal Loss is sued due to class imbalance
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    early_stopping = EarlyStopping(patience=12)

    best_acc = 0.0
    trim_ratio = 0.05 # 5% of highest loss samples are trimmed in each batch

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accs = []

    for epoch in range(num_epochs):

        epoch_start = time.time()

        
        model.train()
        running_train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            
            losses = criterion(outputs, labels)  # shape: (batch_size, num_classes)
            per_sample_loss = losses.sum(dim=1)  # or .mean(dim=1), shape: (batch_size,)

            batch_size = per_sample_loss.shape[0]
            k = int(trim_ratio * batch_size)

            if k > 0:
                trimmed_losses, _ = torch.topk(per_sample_loss, batch_size - k, largest=False)
                loss = trimmed_losses.mean()
            else:
                loss = per_sample_loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_losses.append(epoch_train_loss)


        model.eval()
        running_val_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                val_loss = criterion(outputs, labels).mean()
                running_val_loss += val_loss.item()

                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        epoch_val_loss = running_val_loss / len(test_loader)
        epoch_val_losses.append(epoch_val_loss)

        acc = accuracy_score(targets, preds)
        epoch_val_accs.append(acc)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # For unique filenames every run

    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(base_dir, "plots_convt2") # Directory to save plots
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
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

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
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_convt2_{timestamp}.png"))
    print(f"Saved confusion_matrix_convt2_{timestamp}.png")
    plt.close()
  
  # Sensitivity & Specificity (macro average)

    def safe_div(numerator, denominator):
        return numerator / denominator if denominator != 0 else np.nan

    sensitivity_per_class = np.array([
    safe_div(tp, (tp_fn))
    for tp, tp_fn in zip(np.diag(cm), np.sum(cm, axis=1))])

    specificity_per_class = []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = safe_div(tn, tn + fp)
        specificity_per_class.append(spec)

    macro_sensitivity = np.nanmean(sensitivity_per_class)
    macro_specificity = np.nanmean(specificity_per_class)

    print(f"Macro Sensitivity: {macro_sensitivity:.5f}")
    print(f"Macro Specificity: {macro_specificity:.5f}") 

    # Accuracy
    acc = accuracy_score(all_targets, all_preds)
    print(f"Accuracy: {acc:.5f}")

    # Precision
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    print(f"Macro Precision: {precision:.5f}")

    # F1 Score
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    print(f"Weighted F1 Score: {f1:.5f}")

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    print(f"Macro F1-score: {macro_f1:.5f}")

    # Cohen's Kappa
    kappa = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    print(f"Quadratic Weighted Kappa: {kappa:.5f}")

    # Kurtosis & Entropy (on predicted probabilities)
    kurt = kurtosis(all_probs, axis=None)
    ent = entropy(np.mean(all_probs, axis=0))
    print(f"Kurtosis: {kurt:.5f}")
    print(f"Entropy: {ent:.5f}")

    print(f"Total Training Time: {total_time/60:.2f} minutes")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f} seconds")

    # ROC Curve (One-vs-Rest)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((all_targets == i).astype(int), all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"roc_curve_convt2_{timestamp}.png"))
    print(f"Saved roc_curve_convt2_{timestamp}.png")
    plt.close()
    
    # Box Plot of predicted probabilities
    plt.figure()
    plt.boxplot([all_probs[:, i] for i in range(num_classes)], tick_labels=[f'Class {i}' for i in range(num_classes)], showfliers=False)
    plt.title('Box Plot of Predicted Probabilities')
    plt.ylabel('Probability')
    plt.savefig(os.path.join(plot_dir, f"box_plot_convt2_{timestamp}.png"))
    print(f"Saved box_plot_convt2_{timestamp}.png")
    plt.close()

    # Accuracy vs Epoch and Error vs Epoch
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
    plt.savefig(os.path.join(plot_dir, f"accuracy_loss_vs_epoch_convt2_{timestamp}.png"))
    plt.close()
    print(f"Saved accuracy_loss_vs_epoch_convt2_{timestamp}.png")