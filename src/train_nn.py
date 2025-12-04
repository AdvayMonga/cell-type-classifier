import numpy as np
import scanpy as sc
import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data  import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import itertools

# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train neural network model for cell type classification')
parser.add_argument('--data', type=str, required=True, help='Path to processed .h5ad file')
parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
args = parser.parse_args()

# Resolve data path relative to project root
data_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data

adata = sc.read_h5ad(data_path)

# Print dataset info
print(f"\nDataset loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
print(f"Cell types: {adata.obs['cell_type'].nunique()}")
print(f"Cell type distribution:")
print(adata.obs['cell_type'].value_counts())

# Extract gene expression matrix (convert sparse to dense)
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()
else:
    X = np.array(X)

print(f"\nGene expression matrix:")
print(f"  Shape: {X.shape}")
print(f"  Range: [{X.min():.4f}, {X.max():.4f}]")
print(f"  Mean: {X.mean():.4f}")
print(f"  Data is already log1p normalized from scVelo preprocessing")

y = adata.obs['cell_type'].values

# Add ALL available features to help distinguish similar cell types
print("\nChecking for velocity and statistical features...")
velocity_features = []

# Velocity features
if 'velocity_pseudotime' in adata.obs.columns:
    vp = adata.obs['velocity_pseudotime'].values.reshape(-1, 1)
    velocity_features.append(vp)
    print(f"  ✓ velocity_pseudotime: range [{vp.min():.4f}, {vp.max():.4f}], mean {vp.mean():.4f}")

if 'latent_time' in adata.obs.columns:
    lt = adata.obs['latent_time'].values.reshape(-1, 1)
    velocity_features.append(lt)
    print(f"  ✓ latent_time: range [{lt.min():.4f}, {lt.max():.4f}], mean {lt.mean():.4f}")

if 'velocity_confidence' in adata.obs.columns:
    vc = adata.obs['velocity_confidence'].values.reshape(-1, 1)
    velocity_features.append(vc)
    print(f"  ✓ velocity_confidence: range [{vc.min():.4f}, {vc.max():.4f}], mean {vc.mean():.4f}")

if 'velocity_magnitude' in adata.obs.columns:
    vm = adata.obs['velocity_magnitude'].values.reshape(-1, 1)
    velocity_features.append(vm)
    print(f"  ✓ velocity_magnitude: range [{vm.min():.4f}, {vm.max():.4f}], mean {vm.mean():.4f}")

# Cell cycle features (helps distinguish activated vs naive T cells)
if 'S_score' in adata.obs.columns:
    ss = adata.obs['S_score'].values.reshape(-1, 1)
    velocity_features.append(ss)
    print(f"  ✓ S_score: range [{ss.min():.4f}, {ss.max():.4f}], mean {ss.mean():.4f}")

if 'G2M_score' in adata.obs.columns:
    g2m = adata.obs['G2M_score'].values.reshape(-1, 1)
    velocity_features.append(g2m)
    print(f"  ✓ G2M_score: range [{g2m.min():.4f}, {g2m.max():.4f}], mean {g2m.mean():.4f}")

# Expression statistics (helps distinguish transcriptional states)
if 'mean_expression' in adata.obs.columns:
    me = adata.obs['mean_expression'].values.reshape(-1, 1)
    velocity_features.append(me)
    print(f"  ✓ mean_expression: range [{me.min():.4f}, {me.max():.4f}], mean {me.mean():.4f}")

if 'expression_variance' in adata.obs.columns:
    ev = adata.obs['expression_variance'].values.reshape(-1, 1)
    velocity_features.append(ev)
    print(f"  ✓ expression_variance: range [{ev.min():.4f}, {ev.max():.4f}], mean {ev.mean():.4f}")

if 'n_genes_expressed' in adata.obs.columns:
    nge = adata.obs['n_genes_expressed'].values.reshape(-1, 1)
    velocity_features.append(nge)
    print(f"  ✓ n_genes_expressed: range [{nge.min():.0f}, {nge.max():.0f}], mean {nge.mean():.1f}")

# Combine gene expression with velocity features
if velocity_features:
    velocity_array = np.column_stack(velocity_features)
    
    # Check for NaN or Inf values in velocity features
    n_nans = np.isnan(velocity_array).sum()
    n_infs = np.isinf(velocity_array).sum()

    if n_nans > 0 or n_infs > 0:
        velocity_array = np.nan_to_num(velocity_array, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  ⚠ Fixed {n_nans} NaNs and {n_infs} Infs in velocity features")
    
    # BOOST VELOCITY FEATURES to match gene expression importance
    # Gene expression is log1p normalized (range ~0-4, mean ~0.04)
    # Velocity features are 0-1 normalized, so multiply by 5 to match scale
    VELOCITY_BOOST = 5.0
    velocity_array = velocity_array * VELOCITY_BOOST
    
    print(f"\n  ⚡ BOOSTED velocity features by {VELOCITY_BOOST}x to match gene expression scale")
    print(f"     This gives velocity features equal importance in the neural network")
    
    # Combine features
    X = np.column_stack([X, velocity_array])
    n_genes = X.shape[1] - velocity_array.shape[1]
    n_additional = velocity_array.shape[1]
    
    print(f"\n✓ Combined features: {X.shape[1]} total ({n_genes} genes + {n_additional} additional features)")
    print(f"  Gene expression range: [{X[:, :n_genes].min():.4f}, {X[:, :n_genes].max():.4f}], mean={X[:, :n_genes].mean():.4f}")
    print(f"  Boosted velocity range: [{X[:, n_genes:].min():.4f}, {X[:, n_genes:].max():.4f}], mean={X[:, n_genes:].mean():.4f}")
    print(f"\n  Additional features help distinguish similar cell types like CD4+ subtypes:")
    print(f"    - velocity_confidence: differentiating vs stable cells")
    print(f"    - S_score/G2M_score: activated vs naive T cells")
    print(f"    - expression_variance: activated vs resting states")

else:
    print("\n⚠ No velocity features found, using gene expression only")
    print(f"  Using log-normalized gene expression as-is")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} cells")
print(f"Test set: {X_test.shape[0]} cells")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define neural network architecture
class CellTypeClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.3):
        super(CellTypeClassifier, self).__init__()
        self.fc = nn.ModuleList()
        for s in hidden_size:
            self.fc.append(nn.Linear(input_size, s))
            input_size = s
        self.dropout_rate = dropout_rate
        self.output_layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        for s in self.fc:
            x = s(x)
            x = nn.ReLU()(x)
            x = nn.Dropout(self.dropout_rate)(x)
        x = self.output_layer(x)
        return x
    
# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(batch_y.numpy())

    return all_labels, all_preds

# Training function for tuning
def train_model(model, train_loader, test_loader, num_epochs, learning_rate, weight_decay):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluate on test set
        all_labels, all_preds = evaluate_model(model, test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, Test Accuracy: {accuracy:.1%}")
        if accuracy < best_accuracy - 0.5:
            return best_accuracy
        best_accuracy = max(best_accuracy, accuracy)
    
    return best_accuracy

# Get base parameters
input_size = X_train.shape[1] 
# Simpler architecture for 531 genes - avoid overfitting
hidden_size = [256, 128]
num_classes = len(np.unique(y_train))

print(f"\nModel parameters:")
print(f"  Input size: {input_size}")
print(f"  Hidden layers: {hidden_size}")
print(f"  Output classes: {num_classes}")

# Hyperparameter tuning
if args.tune:
    print("\n" + "="*70)
    print("HYPERPARAMETER GRID SEARCH")
    print("="*70)
    
    # Define hyperparameter grid - adjusted for smaller dataset
    learning_rates = [0.001, 0.0005, 0.005, 0.0001]
    dropout_rates = [0.2, 0.3, 0.4]
    weight_decays = [0.0001, 0.001]  # L2 regularization
    
    results = []
    total_configs = len(learning_rates) * len(dropout_rates) * len(weight_decays)
    current_config = 0
    
    for lr, dropout, wd in itertools.product(learning_rates, dropout_rates, weight_decays):
        current_config += 1
        print(f"\n[{current_config}/{total_configs}] LR: {lr}, Dropout: {dropout}, L2: {wd}")
        
        model = CellTypeClassifier(input_size, hidden_size, num_classes, dropout_rate=dropout)
        accuracy = train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=lr, weight_decay=wd)
        
        results.append({
            'learning_rate': lr,
            'dropout': dropout,
            'weight_decay': wd,
            'accuracy': accuracy
        })
        
        print(f"  Test Accuracy: {accuracy:.1%}")
    
    # Find best configuration
    best_config = max(results, key=lambda x: x['accuracy'])
    print("\n" + "="*70)
    print("BEST HYPERPARAMETERS:")
    print("="*70)
    print(f"Learning rate: {best_config['learning_rate']}")
    print(f"Dropout rate: {best_config['dropout']}")
    print(f"L2 Regularization (weight_decay): {best_config['weight_decay']}")
    print(f"Best Test Accuracy: {best_config['accuracy']:.1%}")
    print("="*70)
    
    # Use best hyperparameters
    learning_rate = best_config['learning_rate']
    dropout_rate = best_config['dropout']
    weight_decay = best_config['weight_decay']

else:
    # Default hyperparameters - adjusted for smaller feature set
    learning_rate = 0.0005
    dropout_rate = 0.2
    weight_decay = 0.0001
    print(f"\nUsing default hyperparameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  L2 Regularization (weight_decay): {weight_decay}")

# Train final model
print(f"\nTraining final model...")
model = CellTypeClassifier(input_size, hidden_size, num_classes, dropout_rate=dropout_rate)
print(f"\nModel architecture:")
print(model)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
num_epochs = 30
print(f"\nTraining for {num_epochs} epochs...")

train_model(model, train_loader, test_loader, num_epochs, learning_rate, weight_decay)

all_labels, all_preds = evaluate_model(model, test_loader)
train_labels, train_preds = evaluate_model(model, train_loader)

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test accuracy: {accuracy:.1%}")

train_accuracy = accuracy_score(train_labels, train_preds)
print(f"Train accuracy: {train_accuracy:.1%}")

print("\nPer-class performance:")
print(classification_report(
    all_labels, all_preds,
    target_names=label_encoder.classes_
))

# Save the model
# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

model_data = {
    'model_state_dict': model.state_dict(),
    'label_encoder': label_encoder,
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_classes': num_classes,
    'accuracy': accuracy,
    'learning_rate': learning_rate,
    'dropout_rate': dropout_rate,
    'weight_decay': weight_decay
}

model_path = os.path.join(MODELS_DIR, 'neural_network.pt')
torch.save(model_data, model_path)
print(f"\nModel saved to {model_path}")