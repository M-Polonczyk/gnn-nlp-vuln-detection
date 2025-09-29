# GNN Models for Vulnerability Detection

This directory contains Graph Neural Network (GNN) models. The models operate on Abstract Syntax Trees (ASTs) and Control Flow Graphs (CFGs) to identify potentially vulnerable code patterns.

## Available Models

### 1. VulnerabilityGCN
**Graph Convolutional Network** with enhancements for vulnerability detection:
- Multiple GCN layers with configurable depth
- Batch normalization for training stability  
- Multiple pooling strategies (mean, max, add, combined)
- Dropout for regularization
- Suitable for most vulnerability detection tasks

### 2. VulnerabilityGAT  
**Graph Attention Network** that learns to focus on important code patterns:
- Multi-head attention mechanism
- Learns which nodes/code elements are most relevant
- Configurable attention heads and concatenation
- Good for complex vulnerability patterns requiring attention

### 3. VulnerabilityGraphSAGE
**GraphSAGE** with hierarchical pooling:
- Scalable to large code graphs
- Inductive learning capability
- TopK pooling for hierarchical analysis
- Attention-based graph-level pooling
- Best for large codebases and scalability

### 4. VulnerabilityGIN
**Graph Isomorphism Network** with multi-level readout:
- Theoretically most expressive for graph structure
- Multi-level feature combination
- Good for detecting structural code patterns
- Suitable when graph topology is crucial

## Deciding if a Model should be separate for every language

TODO: Evaluate if language-specific features significantly impact model performance.

## Quick Start

### Basic Usage

```python
from gnn_vuln_detection.models.gnn import create_vulnerability_detector

# Create a GCN model for binary vulnerability detection
model = create_vulnerability_detector(
    model_type='gcn',
    input_dim=64,        # Dimension of node features
    hidden_dim=128,      # Hidden layer size
    num_classes=2,       # Binary classification
    num_layers=3,        # Number of GNN layers
    dropout_rate=0.3     # Dropout rate
)

# Forward pass with PyTorch Geometric data
output = model(graph_data)  # graph_data should have x, edge_index, batch
predictions = output.argmax(dim=1)
```

### Using Model Factory

```python
from src.gnn_vuln_detection.models.gnn import GNNModelFactory

# Get recommended configurations
configs = GNNModelFactory.get_recommended_config(
    dataset_size=10000, 
    complexity='medium'
)

# Create model with custom config
model = GNNModelFactory.create_model(
    model_type='gat',
    input_dim=64,
    num_classes=2,
    config=configs['gat']
)
```

### Training Example

```python
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# Setup
model = create_vulnerability_detector('gcn', input_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for batch in train_loader:
    optimizer.zero_grad()
    out = model(batch)
    loss = criterion(out, batch.y)
    loss.backward()
    optimizer.step()
```

## Model Configurations

Models can be configured via YAML files in `config/model_params.yml`:

```yaml
gcn_standard:
  model_type: gcn
  hidden_dim: 128
  num_layers: 3
  dropout_rate: 0.3
  use_batch_norm: true
  pool_type: mean
  num_classes: 2

gat_attention:
  model_type: gat  
  hidden_dim: 128
  num_layers: 3
  dropout_rate: 0.3
  heads: 4
  concat_heads: true
```

## Input Data Format

Models expect PyTorch Geometric `Data` objects with:

- `x`: Node features `[num_nodes, num_features]`
- `edge_index`: Edge connectivity `[2, num_edges]`  
- `y`: Graph-level labels `[num_graphs]`
- `batch`: Batch assignment for nodes `[num_nodes]`

### Example Data Creation

```python
import torch
from torch_geometric.data import Data

# Create graph data
x = torch.randn(10, 64)  # 10 nodes, 64 features each
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Edges
y = torch.tensor([1])  # Vulnerable = 1, Safe = 0

graph = Data(x=x, edge_index=edge_index, y=y)
```

## Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| General vulnerability detection | VulnerabilityGCN | Good balance of performance and efficiency |
| Complex patterns requiring attention | VulnerabilityGAT | Attention helps focus on important code |
| Large codebases | VulnerabilityGraphSAGE | Scalable architecture |
| Structural pattern detection | VulnerabilityGIN | Most expressive for graph structure |
| Small datasets | VulnerabilityGCN (lightweight config) | Prevents overfitting |
| Multi-class CWE classification | VulnerabilityGCN (high performance config) | Better capacity for multiple classes |

## Performance Tips

1. **Hidden Dimension**: Start with 128, increase for complex patterns
2. **Number of Layers**: 2-4 layers work best, more may cause over-smoothing
3. **Dropout Rate**: 0.2-0.5 depending on dataset size
4. **Batch Size**: 32-64 for most cases, adjust based on GPU memory
5. **Learning Rate**: Start with 0.001, use scheduler for best results

## Advanced Features

### Combined Pooling
```python
model = create_vulnerability_detector(
    model_type='gcn',
    pool_type='combined'  # Uses both mean and max pooling
)
```

### Multi-Head Attention
```python  
model = create_vulnerability_detector(
    model_type='gat',
    heads=8,              # Number of attention heads
    concat_heads=True     # Concatenate or average heads
)
```

### Hierarchical Analysis
```python
model = create_vulnerability_detector(
    model_type='graphsage'  # Includes TopK pooling
)
```

## Extending the Models

To create custom models, inherit from `BaseGNN`:

```python
from src.gnn_vuln_detection.models.gnn.base import BaseGNN

class CustomVulnModel(BaseGNN):
    def __init__(self, input_dim, hidden_dim, num_classes, **kwargs):
        super().__init__(input_dim, hidden_dim, num_classes, 3, 0.3)
        # Your custom architecture
    
    def forward(self, data):
        # Your custom forward pass
        pass
```

## Integration with Pipeline

These models integrate with the broader vulnerability detection pipeline:

1. **Code → AST/CFG**: Use `code_representation` module
2. **Graph Construction**: Use `graph_builder` 
3. **Feature Extraction**: Use `feature_extractor`
4. **Model Training**: Use these GNN models
5. **Evaluation**: Use `training.metrics`

## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory**: Reduce batch size or hidden dimension
2. **Poor Performance**: Try different model types or increase model capacity
3. **Overfitting**: Increase dropout rate or reduce model complexity
4. **Slow Training**: Use GraphSAGE for large graphs, reduce number of layers

**Debugging:**
```python
# Check model output shapes
print(f"Model output shape: {output.shape}")
print(f"Expected: [batch_size, num_classes]")

# Monitor gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

For more examples, see `examples/gnn_vulnerability_detection_demo.py`.

## Converting Data to PyTorch Geometric Format

### From Dataclasses

The most straightforward way to convert your vulnerability data to PyTorch Geometric format is using dataclasses:

#### 1. Basic CodeSample Conversion

```python
from src.gnn_vuln_detection.data_processing.graph_converter import (
    CodeSample, 
    DataclassToGraphConverter
)

# Create a code sample
sample = CodeSample(
    code="char buf[10]; strcpy(buf, input);",  # Vulnerable code
    label=1,                                   # 1 = vulnerable, 0 = safe
    language="c",
    cwe_id="CWE-120",
    function_name="vulnerable_func"
)

# Convert to PyG Data
converter = DataclassToGraphConverter()
pyg_data = converter.code_sample_to_pyg_data(sample)

# Result: Data(x=[num_nodes, num_features], edge_index=[2, num_edges], y=[1])
print(f"Nodes: {pyg_data.x.shape[0]}")
print(f"Edges: {pyg_data.edge_index.shape[1]}")
print(f"Label: {pyg_data.y.item()}")
```

#### 2. Custom Dataclass Conversion

```python
from dataclasses import dataclass

@dataclass
class VulnerabilityData:
    source_code: str
    is_vulnerable: bool
    severity: float
    cwe_type: str

# Convert custom dataclass to CodeSample
vuln_data = VulnerabilityData(
    source_code="gets(buffer);",
    is_vulnerable=True,
    severity=8.5,
    cwe_type="CWE-120"
)

code_sample = CodeSample(
    code=vuln_data.source_code,
    label=1 if vuln_data.is_vulnerable else 0,
    language="c",
    cwe_id=vuln_data.cwe_type,
    metadata={"severity": vuln_data.severity}
)

pyg_data = converter.code_sample_to_pyg_data(code_sample)
```

### From AST Nodes

Convert Tree-sitter AST nodes directly to PyTorch Geometric format:

```python
from src.gnn_vuln_detection.code_representation.ast_parser import ASTParser
from src.gnn_vuln_detection.data_processing.graph_converter import ASTToGraphConverter

# Parse code to AST
parser = ASTParser("c")
ast_root = parser.parse_code_to_ast("int main() { return 0; }")

# Convert AST to PyG Data
ast_converter = ASTToGraphConverter()
pyg_data = ast_converter.ast_to_pyg_data(
    ast_root, 
    label=0,  # Safe code
    graph_type=GraphType.AST,
    include_edge_features=True
)
```

### Creating Datasets

#### 1. From Multiple Samples

```python
from src.gnn_vuln_detection.data_processing.graph_converter import (
    GraphDataset,
    create_vulnerability_dataset
)

# Create multiple samples
samples = [
    CodeSample("strcpy(buf, input);", 1, "c"),     # Vulnerable
    CodeSample("int x = 5;", 0, "c"),              # Safe
    CodeSample("system(user_cmd);", 1, "c"),       # Vulnerable
    CodeSample("printf('hello');", 0, "c"),        # Safe
]

# Create train/val/test split
train_ds, val_ds, test_ds = create_vulnerability_dataset(
    samples,
    graph_type=GraphType.AST,
    train_split=0.8,
    val_split=0.1
)
```

#### 2. From JSON Dataset

```python
from src.gnn_vuln_detection.data_processing.graph_converter import load_from_json_dataset

# JSON format: [{"code": "...", "label": 1, "cwe_id": "CWE-120"}, ...]
samples = load_from_json_dataset(
    "vulnerability_dataset.json",
    code_field="source_code",
    label_field="is_vulnerable",
    language="c"
)

dataset = GraphDataset(samples)
```

### Data Loading for Training

```python
from torch_geometric.loader import DataLoader

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Use with GNN models
for batch in train_loader:
    # batch.x: [total_nodes_in_batch, num_features]
    # batch.edge_index: [2, total_edges_in_batch]
    # batch.y: [batch_size]
    # batch.batch: [total_nodes_in_batch] - assigns nodes to graphs
    
    output = model(batch)  # Forward pass
    loss = criterion(output, batch.y)
```

### Advanced Graph Types

Convert to different graph representations:

```python
from src.gnn_vuln_detection.code_representation.graph_builder import GraphType

# Abstract Syntax Tree (default)
ast_data = converter.code_sample_to_pyg_data(sample, GraphType.AST)

# Control Flow Graph  
cfg_data = converter.code_sample_to_pyg_data(sample, GraphType.CFG)

# Data Flow Graph
dfg_data = converter.code_sample_to_pyg_data(sample, GraphType.DFG)

# Hybrid graph with multiple edge types
hybrid_data = converter.code_sample_to_pyg_data(
    sample, 
    GraphType.HYBRID,
    include_edge_features=True
)
```

### Expected Data Format

PyTorch Geometric `Data` objects should have:

```python
Data(
    x=torch.tensor([[...], [...]]),        # Node features [num_nodes, num_features]
    edge_index=torch.tensor([[0, 1, 2],    # Source nodes
                             [1, 2, 0]]),  # Target nodes [2, num_edges]
    y=torch.tensor([1]),                   # Graph label [1] 
    batch=torch.tensor([0, 0, 0, 1, 1])   # Node-to-graph assignment [num_nodes]
)
```

### Caching for Performance

Enable caching for large datasets:

```python
dataset = GraphDataset(
    samples, 
    cache_dir="cache/processed_graphs"  # Saves processed graphs to disk
)
```

### Integration Example

Complete example from dataclass to trained model:

```python
# 1. Create data
samples = [CodeSample(...), ...]

# 2. Convert to datasets
train_ds, val_ds, test_ds = create_vulnerability_dataset(samples)

# 3. Create data loaders
train_loader = DataLoader(train_ds, batch_size=32)

# 4. Create model matching data dimensions
sample_data = train_ds[0]
model = create_vulnerability_detector(
    model_type='gcn',
    input_dim=sample_data.x.shape[1],  # Match node feature dimension
    num_classes=2
)

# 5. Train
for batch in train_loader:
    output = model(batch)
    # ... training logic
```
