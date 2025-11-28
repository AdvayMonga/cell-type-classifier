import numpy as np
import scanpy as sc
import os
import pickle
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train logistic regression model for cell type classification')
parser.add_argument('--data', type=str, required=True, help='Path to processed .h5ad file')
args = parser.parse_args()

# Resolve data path relative to project root
data_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data

# Load data
adata = sc.read_h5ad(data_path)

# feature matrix: rows-cells, columns-genes
X = adata.X

# target vector: cell type labels
y = adata.obs['cell_type'].values

# convert to dense array
if hasattr(X, 'toarray'):
    X = X.toarray()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Training logistic regression model on {X.shape[0]} cells and {X.shape[1]} genes...")
# returns training/testing features/labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"tuning hyperparameters...")
# tuning hyperparameters
param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0] # regularization strength
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=5, # cross-validation folds
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest C value: {grid_search.best_params_['C']}")
print(f"Best CV accuracy: {grid_search.best_score_:.1%}")

# make predictions
model = grid_search.best_estimator_
y_pred = model.predict(X_test)
train_pred = model.predict(X_train)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy_train = accuracy_score(y_train, train_pred)
print(f"\nTrain accuracy: {accuracy_train:.1%}")
print(f"Accuracy: {accuracy:.1%}")

# classification report
print("\nPer-class performance:")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_
))

# Save the model
# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

model_data = {
    'model': model,
    'label_encoder': label_encoder,
    'accuracy': accuracy
}

model_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to {model_path}")