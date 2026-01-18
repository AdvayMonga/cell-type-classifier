"""Neural network architecture for cell type classification."""
import torch
import torch.nn as nn


class GeneAttention(nn.Module):
    """Attention mechanism that learns which genes/features to focus on."""

    def __init__(self, input_size, attention_hidden=128):
        super(GeneAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, input_size),
        )

    def forward(self, x):
        """Returns (attended_features, attention_weights)."""
        attention_scores = self.attention(x)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended = x * attention_weights * x.shape[-1]
        return attended, attention_weights


class CellTypeClassifier(nn.Module):
    """MLP with attention for cell type classification."""

    def __init__(self, input_size, num_classes, hidden_sizes=None, dropout_rate=0.3,
                 use_attention=True, attention_hidden=128):
        super(CellTypeClassifier, self).__init__()

        self.use_attention = use_attention
        self.input_size = input_size

        if use_attention:
            self.attention = GeneAttention(input_size, attention_hidden)

        if hidden_sizes is None:
            hidden_sizes = self.get_architecture_for_input_size(input_size)

        self.fc = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.fc.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.dropout_rate = dropout_rate
        self.output_layer = nn.Linear(prev_size, num_classes)
        self._last_attention_weights = None

    def forward(self, x):
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
        """Get attention weights from last forward pass."""
        return self._last_attention_weights

    def get_gene_importance(self, dataloader, device='cpu'):
        """Compute average gene importance across a dataset."""
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
        """Auto-select network architecture based on input dimensions."""
        if input_size > 4000:
            return [512, 256, 128]
        elif input_size > 2000:
            return [256, 128]
        else:
            return [128, 64]
