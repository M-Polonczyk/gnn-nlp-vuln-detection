import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

# PyG
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, HeteroConv, SAGEConv

# Transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

# -----------------------------
# Konfiguracja (możesz dostosować)
# -----------------------------
TASK_WEIGHTS = {
    "cve_detection": 0.3,
    "cwe_classification": 0.25,
    "logic_errors": 0.2,
    "code_duplication": 0.1,
    "complexity_analysis": 0.1,
    "naming_issues": 0.05,
}

HIDDEN_DIM = 256  # wspólny wymiar latent
GNN_HIDDEN = 256
TRANSFORMER_BACKBONE = "microsoft/graphcodebert-base"  # zmień jeśli potrzebujesz
TRANSFORMER_OUT_DIM = 768
DROPOUT = 0.3
NUM_GAT_HEADS = 8


# -----------------------------
# HeteroGNN Encoder
# -----------------------------
class HeteroGNNEncoder(nn.Module):
    """Heterogeniczny GNN encoder budujący reprezentacje grafów.
    Implementacja: HeteroConv z mieszanką GCN / GAT / GraphSAGE dla różnych rodzajów krawędzi.
    Zwraca reprezentację *graph-level* (po pooling) o rozmiarze `out_dim`.
    """

    def __init__(
        self,
        metadata,
        in_dims: dict[str, int],
        out_dim: int = HIDDEN_DIM,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.metadata = metadata  # (node_types, edge_types)
        self.num_layers = num_layers
        self.out_dim = out_dim

        # initial linear per node type -> unify dims
        self.type_proj = nn.ModuleDict()
        for ntype in metadata[0]:
            in_dim = in_dims.get(ntype, out_dim)
            self.type_proj[ntype] = nn.Linear(in_dim, out_dim)

        # stack of heterogeneous conv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            # for each canonical edge type (src, rel, dst) -> pick a conv
            for src, rel, dst in metadata[1]:
                # wybieramy konwolucję w zależności od relacji (heurystyka)
                if "ast" in rel or "syntax" in rel:
                    conv = GCNConv(out_dim, out_dim)
                elif "att" in rel or "attention" in rel or "call" in rel:
                    # używamy GAT dla relacji, gdzie uwaga się przydaje
                    conv = GATConv(
                        out_dim,
                        out_dim // NUM_GAT_HEADS,
                        heads=NUM_GAT_HEADS,
                        concat=False,
                    )
                else:
                    conv = SAGEConv(out_dim, out_dim)
                conv_dict[(src, rel, dst)] = conv
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # post-processing per node type
        self.norms = nn.ModuleDict(
            {ntype: nn.LayerNorm(out_dim) for ntype in metadata[0]},
        )
        self.dropout = nn.Dropout(DROPOUT)

        # attention pooling for graph-level representation
        self.att_pool = nn.ModuleDict(
            {ntype: nn.Sequential(nn.Linear(out_dim, 1)) for ntype in metadata[0]},
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        batch_index_dict: dict[str, torch.Tensor],
    ):
        # x_dict: node_type -> [num_nodes_of_type, feat_dim]
        # edge_index_dict: (src_type, rel, dst_type) -> edge_index
        # batch_index_dict: node_type -> tensor of graph indices per node (for pooling)

        # project features
        h_dict = {}
        for ntype, x in x_dict.items():
            h = self.type_proj[ntype](x)
            h = F.relu(h)
            h_dict[ntype] = h

        # conv layers
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            # postact + norm + drop
            for ntype in h_dict:
                h = F.relu(h_dict[ntype])
                h = self.norms[ntype](h)
                h = self.dropout(h)
                h_dict[ntype] = h

        # attention pooling per node type -> produce graph-level per node type
        graph_parts = []
        for ntype, h in h_dict.items():
            batch_idx = batch_index_dict[ntype]
            # attention scores
            scores = self.att_pool[ntype](h).squeeze(-1)  # [N]
            scores = torch.softmax(
                scores - scores.max(), dim=0,
            )  # numerically stable across nodes
            # compute per-graph pooling via scatter
            # we implement manual scatter mean/weighted-sum
            # note: torch_scatter may be available in your env; for generality, użyj scatter_add
            # batch_idx.scatter_add(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float))
            num_graphs = int(batch_idx.max().item()) + 1
            # weighted sum
            weighted = h * scores.unsqueeze(-1)
            graph_repr = h.new_zeros((num_graphs, h.size(-1)))
            graph_counts = h.new_zeros((num_graphs, 1))
            graph_repr = graph_repr.index_add(0, batch_idx, weighted)
            graph_counts = graph_counts.index_add(0, batch_idx, scores.unsqueeze(-1))
            graph_counts = graph_counts.clamp(min=1e-6)
            graph_repr = graph_repr / graph_counts
            graph_parts.append(graph_repr)

        # concat parts from different node types
        graph_emb = torch.cat(
            graph_parts, dim=-1,
        )  # [num_graphs, out_dim * num_node_types]
        # project to out_dim
        projector = nn.Linear(graph_emb.size(-1), self.out_dim).to(graph_emb.device)
        return projector(graph_emb)


# -----------------------------
# Transformer encoder wrapper
# -----------------------------
class TransformerEncoderWrapper(nn.Module):
    def __init__(
        self,
        backbone_name: str = TRANSFORMER_BACKBONE,
        out_dim: int = HIDDEN_DIM,
        fine_tune_layers: int = 4,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.transformer = AutoModel.from_pretrained(backbone_name)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.transformer_hidden = TRANSFORMER_OUT_DIM
        self.out_dim = out_dim

        # freeze all except last `fine_tune_layers` transformer encoder layers (if supported)
        try:
            # huggingface models expose encoder.layer for bert-like models
            for param in self.transformer.parameters():
                param.requires_grad = False
            # try to unfreeze last layers if possible
            if hasattr(self.transformer, "encoder") and hasattr(
                self.transformer.encoder, "layer",
            ):
                total = len(self.transformer.encoder.layer)
                for i in range(total - fine_tune_layers, total):
                    for p in self.transformer.encoder.layer[i].parameters():
                        p.requires_grad = True
        except Exception:
            # fallback: fine-tune whole model
            for param in self.transformer.parameters():
                param.requires_grad = True

        # projection to latent dim
        self.project = nn.Linear(self.transformer_hidden, out_dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, code_inputs: dict[str, torch.Tensor]):
        # code_inputs: dict as returned by tokenizer (input_ids, attention_mask, ...), batched
        outputs = self.transformer(**code_inputs)
        # take pooled output if exists, otherwise mean pool last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # mean pool last_hidden_state weighted by attention mask
            last = outputs.last_hidden_state  # [B, L, H]
            mask = code_inputs["attention_mask"].unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        projected = self.project(pooled)
        projected = F.relu(projected)
        return self.dropout(projected)


# -----------------------------
# Hybrid Model with Multi-Task Heads
# -----------------------------
class HybridHGNNTransformer(nn.Module):
    def __init__(
        self,
        hetero_metadata,
        node_input_dims: dict[str, int],
        gnn_out_dim: int = HIDDEN_DIM,
        transformer_backbone: str = TRANSFORMER_BACKBONE,
        tasks_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.gnn = HeteroGNNEncoder(
            hetero_metadata, node_input_dims, out_dim=gnn_out_dim,
        )
        self.transformer = TransformerEncoderWrapper(
            backbone_name=transformer_backbone, out_dim=gnn_out_dim,
        )

        # fusion: early or late. tutaj prosta late fusion: concat i MLP
        fusion_in = gnn_out_dim + gnn_out_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, fusion_in),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(fusion_in, gnn_out_dim),
            nn.ReLU(),
        )

        # task heads
        # Primary tasks: cve_detection (binary), cwe_classification (multiclass), logic_errors (binary)
        # Auxiliary tasks: code_duplication (similarity/regression), complexity_analysis (regression), naming_issues (binary)
        self.task_heads = nn.ModuleDict()
        # binary heads -> single logit
        self.task_heads["cve_detection"] = nn.Linear(gnn_out_dim, 1)
        self.task_heads["logic_errors"] = nn.Linear(gnn_out_dim, 1)
        self.task_heads["naming_issues"] = nn.Linear(gnn_out_dim, 1)
        # multi-class CWE -> configurable number of classes
        n_cwe = tasks_config.get("n_cwe", 150) if tasks_config else 150
        self.task_heads["cwe_classification"] = nn.Linear(gnn_out_dim, n_cwe)
        # regression heads
        self.task_heads["complexity_analysis"] = nn.Linear(gnn_out_dim, 1)
        # code_duplication: here we output an embedding for similarity learning
        self.task_heads["code_dup_embedding"] = nn.Linear(gnn_out_dim, gnn_out_dim)

    def forward(
        self,
        hetero_data: HeteroData,
        code_inputs: dict[str, torch.Tensor] | None = None,
    ):
        """
        hetero_data: HeteroData zawiera x_dict, edge_index_dict i batch indexes do pooling
        code_inputs: tokenized code inputs dla transformer (batched)
        zwraca dict task -> logits / embeddings
        """
        # --- GNN part: expect node feature dict + edge_index dict + batch indices per node type
        x_dict = {ntype: hetero_data[ntype].x for ntype in hetero_data.node_types}
        edge_index_dict = {
            etype: hetero_data[etype].edge_index for etype in hetero_data.edge_types
        }
        # batch index per node type -> required for pooling
        batch_index_dict = {}
        for ntype in hetero_data.node_types:
            if hasattr(hetero_data[ntype], "batch"):
                batch_index_dict[ntype] = hetero_data[ntype].batch
            else:
                # jeżeli single-graph per batch i node ordering, tworzymy zeros
                num_nodes = x_dict[ntype].size(0)
                batch_index_dict[ntype] = x_dict[ntype].new_zeros(
                    num_nodes, dtype=torch.long,
                )

        gnn_graph_emb = self.gnn(x_dict, edge_index_dict, batch_index_dict)

        # --- Transformer part
        if code_inputs is not None:
            transformer_emb = self.transformer(code_inputs)
        else:
            # jeżeli brak kodu (np. dataset tylko grafowy) -> zeros
            transformer_emb = gnn_graph_emb.new_zeros(gnn_graph_emb.size())

        # --- Fusion
        fused = torch.cat([gnn_graph_emb, transformer_emb], dim=-1)
        fused = self.fusion_mlp(fused)

        # --- Heads
        out = {}
        out["cve_detection"] = self.task_heads["cve_detection"](fused).squeeze(-1)
        out["logic_errors"] = self.task_heads["logic_errors"](fused).squeeze(-1)
        out["naming_issues"] = self.task_heads["naming_issues"](fused).squeeze(-1)
        out["cwe_classification"] = self.task_heads["cwe_classification"](fused)
        out["complexity_analysis"] = self.task_heads["complexity_analysis"](
            fused,
        ).squeeze(-1)
        out["code_dup_embedding"] = self.task_heads["code_dup_embedding"](fused)

        return out


# -----------------------------
# Loss utils for multi-task
# -----------------------------
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_weights: dict[str, float]) -> None:
        super().__init__()
        self.task_weights = task_weights
        # you can extend to dynamic weighting (GradNorm, uncertainty weighting etc.)

    def forward(self, preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        losses = {}
        total = preds[next(iter(preds))].new_zeros(1)
        # CVE detection (binary)
        if "cve_detection" in preds and "cve_detection" in targets:
            losses["cve_detection"] = F.binary_cross_entropy_with_logits(
                preds["cve_detection"], targets["cve_detection"].float(),
            )
            total = (
                total
                + self.task_weights.get("cve_detection", 0.0) * losses["cve_detection"]
            )
        # logic_errors
        if "logic_errors" in preds and "logic_errors" in targets:
            losses["logic_errors"] = F.binary_cross_entropy_with_logits(
                preds["logic_errors"], targets["logic_errors"].float(),
            )
            total = (
                total
                + self.task_weights.get("logic_errors", 0.0) * losses["logic_errors"]
            )
        # naming_issues
        if "naming_issues" in preds and "naming_issues" in targets:
            losses["naming_issues"] = F.binary_cross_entropy_with_logits(
                preds["naming_issues"], targets["naming_issues"].float(),
            )
            total = (
                total
                + self.task_weights.get("naming_issues", 0.0) * losses["naming_issues"]
            )
        # cwe multiclass
        if "cwe_classification" in preds and "cwe_classification" in targets:
            losses["cwe_classification"] = F.cross_entropy(
                preds["cwe_classification"], targets["cwe_classification"].long(),
            )
            total = (
                total
                + self.task_weights.get("cwe_classification", 0.0)
                * losses["cwe_classification"]
            )
        # complexity regression
        if "complexity_analysis" in preds and "complexity_analysis" in targets:
            losses["complexity_analysis"] = F.mse_loss(
                preds["complexity_analysis"], targets["complexity_analysis"].float(),
            )
            total = (
                total
                + self.task_weights.get("complexity_analysis", 0.0)
                * losses["complexity_analysis"]
            )
        # code duplication: similarity learning - for simplicity, use MSE to target embedding or contrastive loss externally
        if "code_dup_embedding" in preds and "code_dup_target" in targets:
            losses["code_duplication"] = F.mse_loss(
                preds["code_dup_embedding"], targets["code_dup_target"].float(),
            )
            total = (
                total
                + self.task_weights.get("code_duplication", 0.0)
                * losses["code_duplication"]
            )

        return total, losses


# -----------------------------
# Training skeleton
# -----------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    loss_wrapper: MultiTaskLossWrapper,
    device: torch.device,
    scaler=None,
    accum_steps: int = 1,
):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        # batch is a HeteroData with batch info. Move tensors to device
        batch = batch.to(device)
        # prepare code_inputs if available
        code_inputs = None
        if hasattr(batch, "code_tokens"):
            # expectation: batch.code_tokens is dict of input_ids, attention_mask
            code_inputs = {k: v.to(device) for k, v in batch.code_tokens.items()}

        preds = model(batch, code_inputs=code_inputs)
        # prepare targets dict (user needs to attach targets to batch)
        targets = {}
        for tkey in TASK_WEIGHTS:
            if hasattr(batch, tkey):
                targets[tkey] = getattr(batch, tkey).to(device)
        # code_dup_target might be in batch
        if hasattr(batch, "code_dup_target"):
            targets["code_dup_target"] = batch.code_dup_target.to(device)

        loss_val, loss_dict = loss_wrapper(preds, targets)
        loss = loss_val / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss_val.item()

    return running_loss / len(dataloader)


def build_optimizer_scheduler(
    model: nn.Module, lr=1e-3, weight_decay=1e-5, total_steps=10000, warmup_steps=500,
):
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=1,
    )
    return optimizer, scheduler


# -----------------------------
# Utilities / example usage
# -----------------------------
if __name__ == "__main__":
    # Przykładowe metadane hetero: node types i edge types
    node_types = ["token", "ast", "cfg", "pdg", "decl"]
    # canonical edge types: (src, rel_name, dst)
    edge_types = [
        ("ast", "ast_parent", "ast"),
        ("token", "ast_edge", "ast"),
        ("cfg", "cfg_edge", "cfg"),
        ("pdg", "data_flow", "pdg"),
        ("decl", "declares", "token"),
        ("token", "calls", "decl"),
        ("token", "ref", "decl"),
    ]
    metadata = (node_types, edge_types)

    # sample input dims per node type (np. embeddings)
    node_input_dims = dict.fromkeys(node_types, 128)

    model = HybridHGNNTransformer(metadata, node_input_dims)
    print(model)

    # Dalsze kroki:
    # - przygotować pipeline ekstrakcji grafów (AST, CFG, PDG) -> HeteroData
    # - tokenizacja kodu wejściowego dla transformer
    # - utworzyć DataLoader i rozpocząć trening

    print("Model utworzony. Dostosuj loader danych i uruchom trening.")
