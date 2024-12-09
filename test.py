import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from model import PneumoniaClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    classification_report
)
import numpy as np
import seaborn as sns

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model
model = PneumoniaClassifier()
model.load_state_dict(torch.load('model1.pth'))
model = model.to(device)
model.eval()

# Data transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4826, 0.4826, 0.4826], [0.2367, 0.2367, 0.2367])
])

# Load test dataset
test_dataset = datasets.ImageFolder(root='./data/dataset/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Test the model
all_preds = []
all_labels = []
all_scores = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        scores = outputs.squeeze().cpu().numpy()
        preds = (outputs >= 0.5).float()

        all_scores.extend(scores)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_scores = np.array(all_scores)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate metrics
accuracy = np.mean(all_preds == all_labels)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

# Create confusion matrix
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

# Calculate specificity
specificity = tn / (tn + fp)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

# Print all metrics
print("\nDetailed Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall/Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia']))

# Plotting
plt.figure(figsize=(15, 5))

# 1. Confusion Matrix
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 2. ROC Curve
plt.subplot(1, 3, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Add raw numbers to confusion matrix
plt.subplot(1, 3, 3)
labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
values = [tn, fp, fn, tp]
plt.bar(labels, values)
plt.title('Confusion Matrix Values')
plt.xticks(rotation=45)
plt.ylabel('Count')

plt.tight_layout()
plt.show()