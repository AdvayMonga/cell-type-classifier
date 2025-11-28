import numpy as np
import scanpy as sc
import os
import pickle
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Get the project root directory (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train random forest model for cell type classification')
parser.add_argument('--data', type=str, required=True, help='Path to processed .h5ad file')
args = parser.parse_args()

# Resolve data path relative to project root
data_path = os.path.join(PROJECT_ROOT, args.data) if not os.path.isabs(args.data) else args.data

adata = sc.read_h5ad(data_path)

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

print("\nTuning Random Forest hyperparameters...")
param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [20, 30, None],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.1%}")

print("\nEvaluating best model on test set...")
model = grid_search.best_estimator_
y_pred = model.predict(X_test)

#model = RandomForestClassifier(random_state=42, n_jobs=-1, cv=5, n_estimators=200, max_depth=30, min_samples_leaf=2)  # Fixed line
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.1%}")

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
    'best_params': grid_search.best_params_,
    'accuracy': accuracy
}

model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to {model_path}")

with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModel saved to models/random_forest.pkl")