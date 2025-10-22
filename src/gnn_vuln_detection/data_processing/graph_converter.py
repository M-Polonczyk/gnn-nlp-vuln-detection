"""
Converters for transforming ASTs and dataclasses into PyTorch Geometric datasets.

This module provides utilities to convert:
1. Tree-sitter AST nodes to PyG Data objects
2. Dataclasses representing code to PyG Data objects
3. NetworkX graphs to PyG Data objects
4. Collections of any of the above to PyG datasets
"""

from dataclasses import asdict, fields
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from tree_sitter import Node

from gnn_vuln_detection.code_representation.ast_parser import ASTParser
from gnn_vuln_detection.code_representation.code_representation import CodeSample
from gnn_vuln_detection.code_representation.feature_extractor import (
    GraphFeatureExtractor,
)
from gnn_vuln_detection.code_representation.graph_builder import GraphBuilder, GraphType


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
            self.ast_parser = ASTParser(sample.language.value)

        ast_root = self.ast_parser.parse_code_to_ast(sample.code)

        # Convert AST to PyG Data
        data = self.ast_converter.ast_to_pyg_data(
            ast_root,
            sample.label,
            graph_type,
            include_edge_features,
        )

        # Add metadata as additional attributes
        data.cwe_ids = sample.cwe_ids or []
        if sample.function_name:
            data.function_name = sample.function_name
        if sample.metadata:
            for key, value in asdict(sample.metadata).items():
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


# Example usage functions
def example_dataclass_conversion():
    """Example of converting dataclass to PyG Data."""
    from gnn_vuln_detection.dataset import CodeSample
    from src.gnn_vuln_detection.dataset.dataset import LanguageEnum

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
        language=LanguageEnum.C,
        cwe_ids=["CWE-120"],
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


if __name__ == "__main__":
    # Run examples
    print("=== Dataclass Conversion Example ===")
    example_dataclass_conversion()

    print("\n=== AST Conversion Example ===")
    example_ast_conversion()
