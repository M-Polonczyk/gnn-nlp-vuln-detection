import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
)

from .base import BaseGNN


class VulnerabilityGAT(BaseGNN):
    """Graph Attention Network for vulnerability detection"""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_classes=2,
        num_layers=3,
        dropout_rate=0.3,
        heads=4,
        concat_heads=True,
    ) -> None:
        super().__init__(
            input_dim,
            hidden_dim,
            num_classes,
            num_layers,
            dropout_rate,
        )

        self.heads = heads
        self.concat_heads = concat_heads

        # GAT layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(
                input_dim,
                hidden_dim // heads if concat_heads else hidden_dim,
                heads=heads,
                concat=concat_heads,
                dropout=dropout_rate,
            ),
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            in_channels = hidden_dim
            out_channels = hidden_dim // heads if concat_heads else hidden_dim
            self.convs.append(
                GATConv(
                    in_channels,
                    out_channels,
                    heads=heads,
                    concat=concat_heads,
                    dropout=dropout_rate,
                ),
            )

        # Final layer
        if num_layers > 1:
            in_channels = hidden_dim
            self.convs.append(
                GATConv(
                    in_channels,
                    hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout_rate,
                ),
            )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Not the last layer
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        # Classification
        return self.classifier(x)
