"""Training logic for cell type classifier."""
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def train_model(model, train_loader, test_loader, num_epochs=20, 
                learning_rate=0.0001, weight_decay=0.001, patience=3, verbose=True):
    """
    Train model with early stopping.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for Adam optimizer
        weight_decay: L2 regularization strength
        patience: Number of epochs to wait for improvement before stopping
        verbose: Print training progress
    
    Returns:
        float: Best test accuracy achieved
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_test_accuracy = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluation phase
        train_labels, train_preds = evaluate_model(model, train_loader)
        test_labels, test_preds = evaluate_model(model, test_loader)
        
        train_acc = accuracy_score(train_labels, train_preds)
        test_acc = accuracy_score(test_labels, test_preds)
        
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train: {train_acc:.1%}, Test: {test_acc:.1%}")
        
        # Early stopping logic
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_test_accuracy


def evaluate_model(model, data_loader):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
    
    Returns:
        tuple: (labels, predictions) as lists
    """
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
