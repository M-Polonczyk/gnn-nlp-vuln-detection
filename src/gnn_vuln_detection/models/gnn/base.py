import abc

import torch
from numpy import ndarray
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


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

    def _eval_function(self, logits):
        return F.softmax(logits, dim=1), torch.argmax(logits, dim=1)

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

    def evaluate(
        self, data: list[Data] | DataLoader, device="cpu"
    ) -> tuple[ndarray, ndarray, ndarray]:
        self.eval()
        y_true = []
        y_pred_probs = []
        y_pred_labels = []

        with torch.no_grad():
            for batch in data:
                batch = batch.to(device)
                logits = self(batch.x, batch.edge_index, batch.batch)

                # Single-label case
                probs, preds = self._eval_function(logits)

                y_true.append(
                    batch.y.cpu() if isinstance(batch.y, torch.Tensor) else batch.y
                )
                y_pred_probs.append(probs.cpu())
                y_pred_labels.append(preds.cpu())
        avg_labels = torch.cat(y_pred_labels).numpy().sum(axis=1).mean()
        print(f"DEBUG: Średnia liczba etykiet na próbkę: {avg_labels:.2f}")
        return (
            torch.cat(y_true).numpy(),
            torch.cat(y_pred_probs).numpy(),
            torch.cat(y_pred_labels).numpy(),
        )
