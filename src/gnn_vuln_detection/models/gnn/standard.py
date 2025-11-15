import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    BatchNorm,
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from .base import BaseGNN


class VulnerabilityGCN(BaseGNN):
    """Enhanced GCN for vulnerability detection with better architecture"""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_classes=2,
        num_layers=3,
        dropout_rate=0.3,
        use_batch_norm=True,
        pool_type="mean",
    ) -> None:
        super().__init__(
            input_dim,
            hidden_dim,
            num_classes,
            num_layers,
            dropout_rate,
        )

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.pool_type = pool_type

        # GCN layers
        self.convs = nn.ModuleList()
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))

        # Final conv layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))

        # Classifier head
        classifier_input_dim = hidden_dim * 2 if pool_type == "combined" else hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch):
        # # Case 1: single argument -> assume it's a PyG Data object
        # if len(args) == 1:
        #     data = args[0]
        #     if isinstance(data, torch.Tensor):
        #         msg = "Raw Tensor input not supported: GCNConv requires edge_index"
        #         raise RuntimeError(msg)
        #     x, edge_index, batch = data.x, data.edge_index, data.batch

        # # Case 2: three arguments -> assume explicit tensors
        # elif len(args) == 3:
        #     x, edge_index, batch = args
        # else:
        #     msg = f"forward() expected (data) or (x, edge_index, batch), got {len(args)} arguments"
        #     raise TypeError(msg)

        if self.use_batch_norm and len(self.batch_norms) > 0:
            for conv, bn in zip(self.convs, self.batch_norms, strict=False):
                x = conv(x=x, edge_index=edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if len(self.convs) > len(self.batch_norms):
                for i, cconv in enumerate(self.convs):
                    if i >= len(self.batch_norms):
                        x = cconv(x=x, edge_index=edge_index)
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        else:
            for conv in self.convs:
                x = conv(x=x, edge_index=edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Node-level representation
        # for conv, bn in zip(self.convs, getattr(self, "batch_norms", [])):
        #     x = conv(x, edge_index)
        #     x = bn(x)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # # Extra conv layers if any (no batch norm)
        # for i in range(len(getattr(self, "batch_norms", [])), len(self.convs)):
        #     x = self.convs[i](x, edge_index)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # Graph-level pooling
        if self.pool_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool_type == "max":
            x = global_max_pool(x, batch)
        elif self.pool_type == "add":
            x = global_add_pool(x, batch)
        else:  # combined pooling
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
            # Adjust classifier input size for combined pooling
            # if not hasattr(self, "_adjusted_classifier"):
            #     self.classifier[0] = nn.Linear(
            #         self.hidden_dim * 2, self.hidden_dim // 2,
            #     )
            #     self._adjusted_classifier = True

        return self.classifier(x)


class MultilabelGCN(BaseGNN):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_classes=25,
        num_layers=3,
        dropout_rate=0.3,
        use_batch_norm=True,
        pool_type="mean",
    ):
        super().__init__(input_dim, hidden_dim, num_classes, num_layers, dropout_rate)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.pool_type = pool_type
        self.label_threshold = 0.5  # Threshold for multi-label classification

        # GCN layers
        self.convs = nn.ModuleList()
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))

        # Final conv layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))

        # Classifier head
        classifier_input_dim = hidden_dim * 2 if pool_type == "combined" else hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _eval_function(self, logits):
        probs = torch.sigmoid(logits)
        return probs, (probs >= self.label_threshold).float()

    def forward(self, x, edge_index, batch):
        if self.use_batch_norm and len(self.batch_norms) > 0:
            for conv, bn in zip(self.convs, self.batch_norms, strict=False):
                x = conv(x=x, edge_index=edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if len(self.convs) > len(self.batch_norms):
                for i, cconv in enumerate(self.convs):
                    if i >= len(self.batch_norms):
                        x = cconv(x=x, edge_index=edge_index)
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        else:
            for conv in self.convs:
                x = conv(x=x, edge_index=edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if self.pool_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool_type == "max":
            x = global_max_pool(x, batch)
        elif self.pool_type == "add":
            x = global_add_pool(x, batch)
        else:  # combined pooling
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(x)


# TODO: Check and replace some logic
class TODOMultilabelGCN(BaseGNN):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_classes=25,
        num_layers=3,
        dropout_rate=0.3,
        use_batch_norm=True,
        pool_type="mean",
    ):
        super().__init__(input_dim, hidden_dim, num_classes, num_layers, dropout_rate)

        self.use_batch_norm = use_batch_norm
        self.pool_type = pool_type
        self.label_threshold = 0.5

        # convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # first layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if use_batch_norm:
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_dim))

        # classifier
        pooled_dim = hidden_dim * 2 if pool_type == "combined" else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def _eval_function(self, logits):
        probs = torch.sigmoid(logits)
        labels = (probs >= self.label_threshold).float()
        return probs, labels

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_batch_norm:
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if self.pool_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool_type == "max":
            x = global_max_pool(x, batch)
        elif self.pool_type == "add":
            x = global_add_pool(x, batch)
        else:  # combined
            x = torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)],
                dim=1,
            )

        return self.classifier(x)
