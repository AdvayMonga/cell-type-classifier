import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from config import PROCESSED_DATA_PATH


# Load data
adata = sc.read_h5ad(PROCESSED_DATA_PATH)

# feature matrix: rows-cells, columns-genes
X = adata.X

# target vector: cell type labels
y = adata.obs['cell_type'].values

# convert to dense array
if hasattr(X, 'toarray'):
    X = X.toarray()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# returns training/testing features/labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

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

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.1%}")

# classification report
print("\nPer-class performance:")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_
))