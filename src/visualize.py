import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import pickle
import os

# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize cell type classification results')
parser.add_argument('--data', type=str, required=True, help='Path to processed .h5ad file')
args = parser.parse_args()

# Resolve data path relative to project root
data_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data

print("Loading data...")
adata = sc.read_h5ad(data_path)

# Prepare data for all models
X = adata.X
y = adata.obs['cell_type'].values

if hasattr(X, 'toarray'):
    X = X.toarray()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# UMAP
print("\n1. Generating UMAP plot...")
if 'X_umap' not in adata.obsm:
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)

sc.pl.umap(adata, color='cell_type', save='_cell_types.png')
print("   Saved: visualizations/umap_cell_types.png")

# Cell Type Distribution
print("\n2. Generating cell type distribution plot...")
plt.figure(figsize=(10, 6))
cell_counts = adata.obs['cell_type'].value_counts()
plt.bar(range(len(cell_counts)), cell_counts.values)
plt.xticks(range(len(cell_counts)), cell_counts.index, rotation=45, ha='right')
plt.xlabel('Cell Type')
plt.ylabel('Number of Cells')
plt.title('Cell Type Distribution in Dataset')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'cell_type_distribution.png'), dpi=300)
plt.close()
print(f"   Saved: {os.path.join(VIZ_DIR, 'cell_type_distribution.png')}")

# Load all models and get predictions
print("\n3. Loading trained models...")

# Logistic Regression
with open(os.path.join(PROJECT_ROOT, 'models/logistic_regression.pkl'), 'rb') as f:
    logreg_data = pickle.load(f)
    logreg_model = logreg_data['model']
    logreg_pred = logreg_model.predict(X_test)
    logreg_acc = logreg_data['accuracy']

# Random Forest
with open(os.path.join(PROJECT_ROOT, 'models/random_forest.pkl'), 'rb') as f:
    rf_data = pickle.load(f)
    rf_model = rf_data['model']
    rf_pred = rf_model.predict(X_test)
    rf_acc = rf_data['accuracy']

# Neural Network
class CellTypeClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CellTypeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

nn_data = torch.load(os.path.join(PROJECT_ROOT, 'models/neural_network.pt'), weights_only=False)
nn_model = CellTypeClassifier(
    nn_data['input_size'],
    nn_data['hidden_size'],
    nn_data['num_classes']
)
nn_model.load_state_dict(nn_data['model_state_dict'])
nn_model.eval()

X_test_tensor = torch.FloatTensor(X_test)
with torch.no_grad():
    outputs = nn_model(X_test_tensor)
    _, nn_pred = torch.max(outputs, 1)
nn_pred = nn_pred.numpy()
nn_acc = nn_data['accuracy']

print("   All models loaded!")

# Confusion matrices for all 3 models
print("\n4. Generating confusion matrices...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = [
    ('Logistic Regression', logreg_pred, logreg_acc),
    ('Random Forest', rf_pred, rf_acc),
    ('Neural Network', nn_pred, nn_acc)
]

for idx, (name, predictions, accuracy) in enumerate(models):
    cm = confusion_matrix(y_test, predictions)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=axes[idx]
    )
    
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_title(f'{name}\nAccuracy: {accuracy:.1%}')

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'confusion_matrices_comparison.png'), dpi=300)
plt.close()
print(f"   Saved: {os.path.join(VIZ_DIR, 'confusion_matrices_comparison.png')}")

# Model accuracy comparison
print("\n5. Generating accuracy comparison...")

plt.figure(figsize=(10, 6))
model_names = ['Logistic\nRegression', 'Random\nForest', 'Neural\nNetwork']
accuracies = [logreg_acc, rf_acc, nn_acc]

bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylim([0.9, 1.0])
plt.ylabel('Test Accuracy')
plt.title('Model Performance Comparison')

# Add accuracy values on top of bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.1%}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, 'model_accuracy_comparison.png'), dpi=300)
plt.close()
print(f"   Saved: {os.path.join(VIZ_DIR, 'model_accuracy_comparison.png')}")

print("\n" + "="*60)
print("Visualization complete!")
print("="*60)
print("\nGenerated plots:")
print("1. visualizations/umap_cell_types.png")
print("2. visualizations/cell_type_distribution.png")
print("3. visualizations/confusion_matrices_comparison.png")
print("4. visualizations/model_accuracy_comparison.png")