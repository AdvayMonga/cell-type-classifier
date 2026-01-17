"""Neural network architecture for cell type classification."""
import torch
import torch.nn as nn


class GeneAttention(nn.Module):
    """
    Attention mechanism that learns which genes/features to focus on.

    Computes attention weights for each input feature, allowing the model
    to learn which genes are most important for classification.
    """

    def __init__(self, input_size, attention_hidden=128):
        """
        Initialize the attention layer.

        Args:
            input_size: Number of input features (genes)
            attention_hidden: Hidden dimension for attention computation
        """
        super(GeneAttention, self).__init__()

        # Two-layer attention network
        self.attention = nn.Sequential(
            nn.Linear(input_size, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, input_size),
        )

    def forward(self, x):
        """
        Apply attention to input features.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Compute attention scores
        attention_scores = self.attention(x)

        # Apply softmax to get weights that sum to 1
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to input (element-wise multiplication)
        # Scale by input_size to maintain magnitude
        attended = x * attention_weights * x.shape[-1]

        return attended, attention_weights


class CellTypeClassifier(nn.Module):
    """
    Multi-layer perceptron with attention for cell type classification.

    The attention layer learns which genes are most important for classification,
    removing the need for manual gene selection.

    Architecture automatically adapts to input size:
    - Large datasets (5000+ features): [512, 256, 128]
    - Medium datasets (2000-5000 features): [256, 128]
    - Small datasets (<2000 features): [128, 64]
    """

    def __init__(self, input_size, num_classes, hidden_sizes=None, dropout_rate=0.3,
                 use_attention=True, attention_hidden=128):
        """
        Initialize the classifier.

        Args:
            input_size: Number of input features
            num_classes: Number of output classes
            hidden_sizes: List of hidden layer sizes (auto-selected if None)
            dropout_rate: Dropout probability (default 0.3)
            use_attention: Whether to use attention mechanism (default True)
            attention_hidden: Hidden dimension for attention layer
        """
        super(CellTypeClassifier, self).__init__()

        self.use_attention = use_attention
        self.input_size = input_size

        # Attention layer for learning gene importance
        if use_attention:
            self.attention = GeneAttention(input_size, attention_hidden)

        # default architecture based on input size
        if hidden_sizes is None:
            hidden_sizes = self.get_architecture_for_input_size(input_size)

        self.fc = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.fc.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.dropout_rate = dropout_rate
        self.output_layer = nn.Linear(prev_size, num_classes)

        # Store attention weights for interpretation
        self._last_attention_weights = None

    def forward(self, x):
        """Forward pass through the network."""
        # Apply attention if enabled
        if self.use_attention:
            x, attention_weights = self.attention(x)
            self._last_attention_weights = attention_weights

        for layer in self.fc:
            x = layer(x)
            x = nn.ReLU()(x)
            x = nn.Dropout(self.dropout_rate)(x)
        x = self.output_layer(x)
        return x

    def get_attention_weights(self):
        """
        Get the attention weights from the last forward pass.

        Returns:
            Tensor of shape (batch_size, input_size) with attention weights,
            or None if attention is disabled or no forward pass has been done.
        """
        return self._last_attention_weights

    def get_gene_importance(self, dataloader, device='cpu'):
        """
        Compute average gene importance across a dataset.

        Args:
            dataloader: DataLoader with input data
            device: Device to run computation on

        Returns:
            Numpy array of shape (input_size,) with average attention weights
        """
        if not self.use_attention:
            return None

        self.eval()
        total_weights = torch.zeros(self.input_size, device=device)
        n_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                _ = self.forward(x)
                total_weights += self._last_attention_weights.sum(dim=0)
                n_samples += x.shape[0]

        avg_weights = total_weights / n_samples
        return avg_weights.cpu().numpy()
    
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
            return [512, 256, 128]
        elif input_size > 2000:
            return [256, 128]
        else:
            return [128, 64]
