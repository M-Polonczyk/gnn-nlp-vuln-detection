"""
Converters for transforming ASTs and dataclasses into PyTorch Geometric datasets.

This module provides utilities to convert:
1. Tree-sitter AST nodes to PyG Data objects
2. Dataclasses representing code to PyG Data objects
3. NetworkX graphs to PyG Data objects
4. Collections of any of the above to PyG datasets
"""

import json
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from gnn_vuln_detection.code_representation.ast_parser import ASTParser
from gnn_vuln_detection.code_representation.code_representation import CodeSample
from gnn_vuln_detection.code_representation.feature_extractor import (
    CodeGraphProcessor,
)
from gnn_vuln_detection.code_representation.graph_builder import GraphBuilder, GraphType
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from tree_sitter import Node


class ASTToGraphConverter:
    """Converts AST nodes to graph representations suitable for GNNs."""

    # Number of fixed numerical features (children, start/end line/col, text_len, is_named)
    NUM_NUMERICAL_FEATURES = 7

    def __init__(self, feature_extractor: CodeGraphProcessor | None = None) -> None:
        self.feature_extractor = feature_extractor or CodeGraphProcessor()
        self.graph_builder = GraphBuilder()
        self.node_type_vocab: dict[str, int] = {}
        self.edge_type_vocab: dict[str, int] = {}
        self._vocab_frozen = False
        self._num_extra_features = 0  # Features from GraphFeatureExtractor

    def ast_to_networkx(
        self,
        ast_node: Node,
        graph_type: GraphType = GraphType.AST,
    ) -> nx.DiGraph:
        """Convert AST node to NetworkX graph."""
        return self.graph_builder.build_graph(ast_node, graph_type)

    def build_vocabulary(self, ast_node: Node) -> None:
        """
        Build vocabulary by traversing AST and collecting all node types.

        Call this method on all samples BEFORE calling extract_node_features
        to ensure consistent feature dimensions across all samples.
        """
        if self._vocab_frozen:
            return

        def _traverse(node: Node) -> None:
            node_type = node.type
            if node_type not in self.node_type_vocab:
                self.node_type_vocab[node_type] = len(self.node_type_vocab)
            for child in node.children:
                _traverse(child)

        _traverse(ast_node)

    def freeze_vocabulary(self) -> None:
        """
        Freeze the vocabulary after building it.

        After freezing, new node types will be mapped to a special <UNK> token.
        """
        if not self._vocab_frozen:
            # Add unknown token for unseen node types
            if "<UNK>" not in self.node_type_vocab:
                self.node_type_vocab["<UNK>"] = len(self.node_type_vocab)
            self._vocab_frozen = True

    def get_num_node_types(self) -> int:
        """Return the number of unique node types in vocabulary."""
        return len(self.node_type_vocab)

    def get_input_dim(self) -> int:
        """
        Calculate and return the input dimension for the GNN model.

        input_dim = num_node_types (OHE) + NUM_NUMERICAL_FEATURES + num_extra_features
        """
        return (
            self.get_num_node_types()
            + self.NUM_NUMERICAL_FEATURES
            + self._num_extra_features
        )

    def set_num_extra_features(self, num_features: int) -> None:
        """Set the number of extra features from GraphFeatureExtractor."""
        self._num_extra_features = num_features

    def extract_node_features(self, node: Node) -> np.ndarray:
        """
        Extract features for a single AST node.

        Feature vector structure:
        - One-hot encoded node type (size: num_node_types)
        - Numerical features (size: NUM_NUMERICAL_FEATURES = 7)
        - Extra features from GraphFeatureExtractor (size: num_extra_features)

        Returns:
            np.ndarray of shape (input_dim,)
        """
        # Ensure vocabulary is frozen for consistent dimensions
        if not self._vocab_frozen:
            self.freeze_vocabulary()

        num_node_types = self.get_num_node_types()
        features = []

        # Node type (one-hot encoded with fixed size)
        node_type = node.type
        if node_type in self.node_type_vocab:
            type_idx = self.node_type_vocab[node_type]
        else:
            # Map unknown types to <UNK> token
            type_idx = self.node_type_vocab.get("<UNK>", 0)

        node_type_one_hot = np.zeros(num_node_types, dtype=np.float32)
        node_type_one_hot[type_idx] = 1.0
        features.extend(node_type_one_hot.tolist())

        # Node properties (7 numerical features)
        features.extend(
            [
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

    def save_vocabulary(self, path: str | Path) -> None:
        """
        Save vocabulary to a JSON file for later use.

        Args:
            path: Path to save the vocabulary file.
        """
        path = Path(path)
        vocab_data = {
            "node_type_vocab": self.node_type_vocab,
            "edge_type_vocab": self.edge_type_vocab,
            "vocab_frozen": self._vocab_frozen,
            "num_extra_features": self._num_extra_features,
            "num_numerical_features": self.NUM_NUMERICAL_FEATURES,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

    def load_vocabulary(self, path: str | Path) -> None:
        """
        Load vocabulary from a JSON file.

        Args:
            path: Path to the vocabulary file.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.node_type_vocab = vocab_data["node_type_vocab"]
        self.edge_type_vocab = vocab_data["edge_type_vocab"]
        self._vocab_frozen = vocab_data.get("vocab_frozen", True)
        self._num_extra_features = vocab_data.get("num_extra_features", 0)

    def ast_to_pyg_data(
        self,
        ast_node: Node,
        labels: list[int],
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
        y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)
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
            "y": y,
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

        ast_root = self.ast_parser.parse_code_to_ast(
            self.ast_parser.cleanup_code(sample.code)
        )

        # Convert AST to PyG Data
        data = self.ast_converter.ast_to_pyg_data(
            ast_root,
            sample.cwe_ids_labeled or [0 for _ in range(9)],  # TODO: make dynamic
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


def networkx_to_pyg_data(
    nx_graph: nx.Graph,
    node_features: np.ndarray,
    labels: list[int],
    include_edge_features: bool = False,
) -> Data:
    """Convert NetworkX graph to PyTorch Geometric Data object."""
    pyg_data = from_networkx(nx_graph)

    # Set node features
    pyg_data.x = torch.tensor(node_features, dtype=torch.float)

    # Set labels
    pyg_data.y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)

    # Optionally set edge features
    if include_edge_features and "edge_attr" in pyg_data:
        pyg_data.edge_attr = torch.tensor(
            pyg_data.edge_attr,
            dtype=torch.float,
        )

    return pyg_data
