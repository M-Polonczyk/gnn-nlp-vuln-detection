import abc

import torch
import torch.nn.functional as F
from torch import nn


class BaseGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Tutaj zdefiniujesz warstwy, które będą wspólne lub nadpisane
        # przez konkretne implementacje GNN

    @abc.abstractmethod
    def forward(self, data) -> None:
        # Data to zazwyczaj obiekt PyTorch Geometric Data lub Batch
        # zawierający x (cechy węzłów), edge_index (indeksy krawędzi),
        # edge_attr (cechy krawędzi), batch (indeksy batcha dla węzłów)
        x, edge_index = data.x, data.edge_index

    def reset_parameters(self) -> None:
        """Weights initialization for all layers."""
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def evaluate(self, data, device="cpu"):
        self.eval()
        y_true = []
        y_pred_probs = []
        y_pred_labels = []

        with torch.no_grad():
            for batch in data:
                batch = batch.to(device)
                logits = self(batch.x, batch.edge_index, batch.batch)

                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                y_true.append(batch.y.cpu())
                y_pred_probs.append(probs.cpu())
                y_pred_labels.append(preds.cpu())

        y_true = torch.cat(y_true).numpy()
        y_pred_probs = torch.cat(y_pred_probs).numpy()
        y_pred_labels = torch.cat(y_pred_labels).numpy()

        return y_true, y_pred_probs, y_pred_labels
