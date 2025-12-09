"""Neural network architecture for cell type classification."""
import torch
import torch.nn as nn


class CellTypeClassifier(nn.Module):
    """
    Multi-layer perceptron for cell type classification.
    
    Architecture automatically adapts to input size:
    - Large datasets (5000+ features): [512, 256, 128]
    - Medium datasets (2000-5000 features): [256, 128]
    - Small datasets (<2000 features): [128, 64]
    """
    
    def __init__(self, input_size, num_classes, hidden_sizes=None, dropout_rate=0.3):
        """
        Initialize the classifier.
        
        Args:
            input_size: Number of input features
            num_classes: Number of output classes
            hidden_sizes: List of hidden layer sizes (auto-selected if None)
            dropout_rate: Dropout probability (default 0.3)
        """
        super(CellTypeClassifier, self).__init__()
        
        # Auto-select architecture if not provided
        if hidden_sizes is None:
            hidden_sizes = self.get_architecture_for_input_size(input_size)
        
        self.fc = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.fc.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.dropout_rate = dropout_rate
        self.output_layer = nn.Linear(prev_size, num_classes)
    
    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.fc:
            x = layer(x)
            x = nn.ReLU()(x)
            x = nn.Dropout(self.dropout_rate)(x)
        x = self.output_layer(x)
        return x
    
    @staticmethod
    def get_architecture_for_input_size(input_size):
        """
        Auto-select network architecture based on input dimensions.
        
        Args:
            input_size: Number of input features
        
        Returns:
            List of hidden layer sizes
        """
        if input_size > 4000:
            return [512, 256, 128]  # Large network for 5000+ features
        elif input_size > 2000:
            return [256, 128]  # Medium network
        else:
            return [128, 64]  # Small network
