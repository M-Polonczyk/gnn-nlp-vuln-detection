"""
Converters for transforming ASTs and dataclasses into PyTorch Geometric datasets.

This module provides utilities to convert:
1. Tree-sitter AST nodes to PyG Data objects
2. Dataclasses representing code to PyG Data objects
3. NetworkX graphs to PyG Data objects
4. Collections of any of the above to PyG datasets
"""

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.utils import from_networkx
from tree_sitter import Node

from gnn_vuln_detection.code_representation.ast_parser import ASTParser
from gnn_vuln_detection.code_representation.feature_extractor import (
    GraphFeatureExtractor,
)
from gnn_vuln_detection.code_representation.graph_builder import GraphBuilder, GraphType


@dataclass
class CodeSample:
    """Dataclass representing a code sample for vulnerability detection."""

    code: str
    label: int  # 0: safe, 1: vulnerable
    language: str = "c"
    cwe_id: str | None = None
    file_path: str | None = None
    function_name: str | None = None
    line_numbers: tuple[int, int] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class GraphFeatures:
    """Dataclass for extracted graph features."""

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray | None = None
    node_types: list[str] | None = None
    edge_types: list[str] | None = None
    global_features: np.ndarray | None = None


class ASTToGraphConverter:
    """Converts AST nodes to graph representations suitable for GNNs."""

    def __init__(self, feature_extractor: GraphFeatureExtractor | None = None) -> None:
        self.feature_extractor = feature_extractor or GraphFeatureExtractor()
        self.graph_builder = GraphBuilder()
        self.node_type_vocab = {}
        self.edge_type_vocab = {}

    def ast_to_networkx(
        self,
        ast_node: Node,
        graph_type: GraphType = GraphType.AST,
    ) -> nx.DiGraph:
        """Convert AST node to NetworkX graph."""
        return self.graph_builder.build_graph(ast_node, graph_type)

    def extract_node_features(self, node: Node) -> np.ndarray:
        """Extract features for a single AST node."""
        # Basic node features
        features = []

        # Node type (one-hot encoded)
        node_type = node.type
        if node_type not in self.node_type_vocab:
            self.node_type_vocab[node_type] = len(self.node_type_vocab)
        type_idx = self.node_type_vocab[node_type]

        # Node properties
        features.extend(
            [
                type_idx,
                len(node.children),  # Number of children
                node.start_point[0],  # Start line
                node.start_point[1],  # Start column
                node.end_point[0],  # End line
                node.end_point[1],  # End column
                len(node.text) if node.text else 0,  # Text length
                1 if node.is_named else 0,  # Is named node
            ],
        )

        # Text-based features if available
        if hasattr(self.feature_extractor, "extract_node_features"):
            text_features = self.feature_extractor.extract_node_features(node)
            features.extend(text_features)

        return np.array(features, dtype=np.float32)

    def ast_to_pyg_data(
        self,
        ast_node: Node,
        label: int,
        graph_type: GraphType = GraphType.AST,
        include_edge_features: bool = False,
    ) -> Data:
        """Convert AST node directly to PyTorch Geometric Data object."""

        # Build NetworkX graph
        nx_graph = self.ast_to_networkx(ast_node, graph_type)

        # Extract node features
        node_features = []
        node_mapping = {}  # Map NetworkX node IDs to tensor indices

        for i, (node_id, node_data) in enumerate(nx_graph.nodes(data=True)):
            node_mapping[node_id] = i
            if "ast_node" in node_data:
                features = self.extract_node_features(node_data["ast_node"])
            else:
                # Default features if no AST node available
                features = np.zeros(8, dtype=np.float32)
                features[0] = node_id  # Use node ID as primary feature
            node_features.append(features)

        # Convert to tensors
        x = torch.tensor(np.array(node_features), dtype=torch.float)

        # Extract edge information
        edge_list = []
        edge_features = []

        for src, dst, edge_data in nx_graph.edges(data=True):
            edge_list.append([node_mapping[src], node_mapping[dst]])

            if include_edge_features:
                edge_type = edge_data.get("type", "ast")
                if edge_type not in self.edge_type_vocab:
                    self.edge_type_vocab[edge_type] = len(self.edge_type_vocab)
                edge_features.append([self.edge_type_vocab[edge_type]])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Create Data object
        data_dict = {
            "x": x,
            "edge_index": edge_index,
            "y": torch.tensor([label], dtype=torch.long),
        }

        if include_edge_features and edge_features:
            data_dict["edge_attr"] = torch.tensor(edge_features, dtype=torch.float)

        return Data(**data_dict)


class DataclassToGraphConverter:
    """Converts dataclasses to PyTorch Geometric Data objects."""

    def __init__(self) -> None:
        self.ast_parser = ASTParser()
        self.ast_converter = ASTToGraphConverter()

    def code_sample_to_pyg_data(
        self,
        sample: CodeSample,
        graph_type: GraphType = GraphType.AST,
        include_edge_features: bool = False,
    ) -> Data:
        """Convert CodeSample dataclass to PyG Data object."""

        # Parse code to AST
        if sample.language != self.ast_parser.language:
            self.ast_parser = ASTParser(sample.language)

        ast_root = self.ast_parser.parse_code_to_ast(sample.code)

        # Convert AST to PyG Data
        data = self.ast_converter.ast_to_pyg_data(
            ast_root,
            sample.label,
            graph_type,
            include_edge_features,
        )

        # Add metadata as additional attributes
        if sample.cwe_id:
            data.cwe_id = sample.cwe_id
        if sample.file_path:
            data.file_path = sample.file_path
        if sample.function_name:
            data.function_name = sample.function_name
        if sample.metadata:
            for key, value in sample.metadata.items():
                setattr(data, key, value)

        return data

    def dataclass_to_features(self, obj: Any) -> dict[str, Any]:
        """Extract features from any dataclass."""
        if not hasattr(obj, "__dataclass_fields__"):
            msg = "Object must be a dataclass"
            raise ValueError(msg)

        features = {}
        for field in fields(obj):
            value = getattr(obj, field.name)

            # Convert different types to tensors
            if isinstance(value, (int, float)):
                features[field.name] = torch.tensor([value], dtype=torch.float)
            elif isinstance(value, str):
                # Simple string encoding (could be enhanced with embeddings)
                features[field.name] = torch.tensor(
                    [hash(value) % 10000],
                    dtype=torch.float,
                )
            elif isinstance(value, (list, tuple)):
                if len(value) > 0 and isinstance(value[0], (int, float)):
                    features[field.name] = torch.tensor(value, dtype=torch.float)

        return features


def load_from_json_dataset(
    json_file: str,
    code_field: str = "code",
    label_field: str = "label",
    language: str = "c",
) -> list[CodeSample]:
    """Load CodeSample objects from JSON dataset.

    Args:
        json_file: Path to JSON file
        code_field: Field name containing code
        label_field: Field name containing label
        language: Programming language

    Returns:
        List of CodeSample objects
    """
    import json

    with open(json_file) as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = CodeSample(
            code=item[code_field],
            label=item[label_field],
            language=language,
            cwe_id=item.get("cwe_id"),
            file_path=item.get("file_path"),
            function_name=item.get("function_name"),
            metadata={
                k: v
                for k, v in item.items()
                if k
                not in [code_field, label_field, "cwe_id", "file_path", "function_name"]
            },
        )
        samples.append(sample)

    return samples


# TODO: Move to dataset.py

class VulnerabilityDataset(Dataset):
    """PyTorch Geometric Dataset for vulnerability detection."""

    def __init__(
        self,
        samples: list[CodeSample | Node | nx.DiGraph],
        root: str | None = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        graph_type: GraphType = GraphType.AST,
        include_edge_features: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        self.samples = samples
        self.graph_type = graph_type
        self.include_edge_features = include_edge_features
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Initialize converters
        self.dataclass_converter = DataclassToGraphConverter()
        self.ast_converter = ASTToGraphConverter()

        super().__init__(root, transform, pre_transform, pre_filter)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def raw_file_names(self):
        return ["some_file_1", "some_file_2", ...]

    @property
    def processed_file_names(self):
        return ["data_1.pt", "data_2.pt", ...]

    def download(self):
        path = download_url("", self.raw_dir)
        return path

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:
        """Get a single data sample."""

        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"sample_{idx}.pt"
            if cache_file.exists():
                return torch.load(cache_file)

        sample = self.samples[idx]

        # Convert based on sample type
        if isinstance(sample, CodeSample):
            data = self.dataclass_converter.code_sample_to_pyg_data(
                sample,
                self.graph_type,
                self.include_edge_features,
            )
        elif isinstance(sample, Node):  # Tree-sitter Node
            # Assume label 0 for unlabeled samples
            data = self.ast_converter.ast_to_pyg_data(
                sample,
                0,
                self.graph_type,
                self.include_edge_features,
            )
        elif isinstance(sample, nx.DiGraph):
            # Convert NetworkX graph to PyG Data
            data = from_networkx(sample)
            if not hasattr(data, "y"):
                data.y = torch.tensor([0], dtype=torch.long)
        else:
            msg = f"Unsupported sample type: {type(sample)}"
            raise ValueError(msg)

        # Cache the result
        if self.cache_dir:
            cache_file = self.cache_dir / f"sample_{idx}.pt"
            torch.save(data, cache_file)

        return data



def create_vulnerability_dataset(
    code_samples: list[CodeSample],
    graph_type: GraphType = GraphType.AST,
    include_edge_features: bool = False,
    cache_dir: str | None = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> tuple[VulnerabilityDataset, VulnerabilityDataset, VulnerabilityDataset]:
    """Create train/val/test datasets from code samples.

    Args:
        code_samples: List of CodeSample objects
        graph_type: Type of graph to build
        include_edge_features: Whether to include edge features
        cache_dir: Directory to cache processed data
        train_split: Fraction for training set
        val_split: Fraction for validation set

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """

    # Shuffle samples
    import random

    samples = code_samples.copy()
    random.shuffle(samples)

    # Split data
    n_total = len(samples)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    # Create datasets
    train_dataset = VulnerabilityDataset(
        train_samples,
        graph_type=graph_type,
        include_edge_features=include_edge_features,
        cache_dir=f"{cache_dir}/train" if cache_dir else None,
    )

    val_dataset = VulnerabilityDataset(
        val_samples,
        graph_type=graph_type,
        include_edge_features=include_edge_features,
        cache_dir=f"{cache_dir}/val" if cache_dir else None,
    )

    test_dataset = VulnerabilityDataset(
        test_samples,
        graph_type=graph_type,
        include_edge_features=include_edge_features,
        cache_dir=f"{cache_dir}/test" if cache_dir else None,
    )

    return train_dataset, val_dataset, test_dataset


# Example usage functions
def example_dataclass_conversion():
    """Example of converting dataclass to PyG Data."""

    # Create sample data
    sample = CodeSample(
        code="""
        int vulnerable_function(char *input) {
            char buffer[10];
            strcpy(buffer, input);  // Buffer overflow vulnerability
            return strlen(buffer);
        }
        """,
        label=1,  # Vulnerable
        language="c",
        cwe_id="CWE-120",
        function_name="vulnerable_function",
    )

    # Convert to PyG Data
    converter = DataclassToGraphConverter()
    data = converter.code_sample_to_pyg_data(sample, GraphType.AST)

    print(f"Node features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Label: {data.y.item()}")
    print(f"CWE ID: {data.cwe_id}")

    return data


def example_ast_conversion():
    """Example of converting AST directly to PyG Data."""

    code = """
    int safe_function(char *input, size_t max_len) {
        char buffer[10];
        strncpy(buffer, input, max_len);  // Safe version
        buffer[9] = '\\0';  // Ensure null termination
        return strlen(buffer);
    }
    """

    # Parse to AST
    parser = ASTParser("c")
    ast_root = parser.parse_code_to_ast(code)

    # Convert to PyG Data
    converter = ASTToGraphConverter()
    data = converter.ast_to_pyg_data(ast_root, label=0)  # Safe code

    print(f"Node features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Label: {data.y.item()}")

    return data


def example_dataset_creation():
    """Example of creating a complete dataset."""

    # Create sample data
    samples = [
        CodeSample("int x = 5;", 0, "c"),
        CodeSample("char buf[10]; strcpy(buf, long_string);", 1, "c"),
        CodeSample("int safe_add(int a, int b) { return a + b; }", 0, "c"),
    ]

    # Create datasets
    train_ds, val_ds, test_ds = create_vulnerability_dataset(
        samples,
        graph_type=GraphType.AST,
        cache_dir="cache/vulnerability_detection",
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Load a sample
    sample_data = train_ds[0]
    print(f"Sample shape: {sample_data.x.shape}")

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Run examples
    print("=== Dataclass Conversion Example ===")
    example_dataclass_conversion()

    print("\n=== AST Conversion Example ===")
    example_ast_conversion()

    print("\n=== Dataset Creation Example ===")
    example_dataset_creation()
