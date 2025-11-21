import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from config import PROCESSED_DATA_PATH

adata = sc.read_h5ad(PROCESSED_DATA_PATH)

X = adata.X
y = adata.obs['cell_type'].values

if hasattr(X, 'toarray'):
    X = X.toarray()

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
    
# Initialize  model
input_size = X_train.shape[1] 
hidden_size = 128
num_classes = len(np.unique(y_train))

model = CellTypeClassifier(input_size, hidden_size, num_classes)
print(f"\nModel architecture:")
print(model)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
# make prediction, measure error, update weights, repeat
num_epochs = 50
print(f"\nTraining for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation
print("\nEvaluating on test set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test accuracy: {accuracy:.1%}")

print("\nPer-class performance:")
print(classification_report(
    all_labels, all_preds,
    target_names=label_encoder.classes_
))

# Evaluation
print("\nEvaluating on test set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test accuracy: {accuracy:.1%}")

print("\nPer-class performance:")
print(classification_report(
    all_labels, all_preds,
    target_names=label_encoder.classes_
))