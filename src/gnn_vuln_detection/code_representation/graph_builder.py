"""Conversion of AST to various graph types (CFG, DFG, etc.)"""

from enum import Enum
from typing import Any

import networkx as nx
import torch
from tree_sitter import Node


class GraphType(Enum):
    """Types of graphs that can be built from AST"""

    AST = "ast"
    CFG = "cfg"  # Control Flow Graph
    DFG = "dfg"  # Data Flow Graph
    PDG = "pdg"  # Program Dependence Graph
    HYBRID = "hybrid"  # Combined graph with multiple edge types


class EdgeType(Enum):
    """Types of edges in the graph"""

    AST_EDGE = "ast"
    CONTROL_FLOW = "control_flow"
    DATA_FLOW = "data_flow"
    CALL = "call"
    RETURN = "return"
    NEXT_TOKEN = "next_token"


class GraphBuilder:
    """Builds different types of graphs from AST nodes"""

    def __init__(self) -> None:
        self.node_counter = 0
        self.variable_definitions: dict[str, list[int]] = {}
        self.variable_uses: dict[str, list[int]] = {}

    def build_graph(
        self,
        ast_root: Node,
        graph_type: GraphType = GraphType.AST,
    ) -> nx.DiGraph:
        """Build a graph from AST root node

        Args:
            ast_root: Root node of the AST
            graph_type: Type of graph to build

        Returns:
            NetworkX directed graph
        """
        self.node_counter = 0
        self.variable_definitions.clear()
        self.variable_uses.clear()

        if graph_type == GraphType.AST:
            return self._build_ast_graph(ast_root)
        if graph_type == GraphType.CFG:
            return self._build_cfg_graph(ast_root)
        if graph_type == GraphType.DFG:
            return self._build_dfg_graph(ast_root)
        if graph_type == GraphType.PDG:
            return self._build_pdg_graph(ast_root)
        if graph_type == GraphType.HYBRID:
            return self._build_hybrid_graph(ast_root)
        msg = f"Unsupported graph type: {graph_type}"
        raise ValueError(msg)

    def _build_ast_graph(self, ast_root: Node) -> nx.DiGraph:
        """Build AST graph preserving tree structure"""
        graph = nx.DiGraph()

        def traverse_ast(node: Node, parent_id: int | None = None) -> int:
            node_id = self.node_counter
            self.node_counter += 1

            # Add node with attributes
            graph.add_node(node_id, **self._extract_node_features(node))

            # Add edge from parent if exists
            if parent_id is not None:
                graph.add_edge(parent_id, node_id, edge_type=EdgeType.AST_EDGE.value)

            # Recursively process children
            for child in node.children:
                traverse_ast(child, node_id)

            return node_id

        traverse_ast(ast_root)
        return graph

    def _extract_node_features(self, node: Node) -> dict[str, Any]:
        """Extract features from a tree-sitter node"""
        text = node.text.decode("utf-8") if node.text else ""

        return {
            "node_type": node.type,
            "text": text,
            "start_point": node.start_point,
            "end_point": node.end_point,
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "is_named": node.is_named,
            "has_error": node.has_error,
            "child_count": len(node.children),
            "text_length": len(text),
            "is_leaf": len(node.children) == 0,
        }

    def _build_cfg_graph(self, ast_root: Node) -> nx.DiGraph:
        """Build Control Flow Graph - simplified implementation"""
        # For now, return AST graph as CFG approximation
        return self._build_ast_graph(ast_root)

    def _build_dfg_graph(self, ast_root: Node) -> nx.DiGraph:
        """Build Data Flow Graph - simplified implementation"""
        # For now, return AST graph as DFG approximation
        return self._build_ast_graph(ast_root)

    def _build_pdg_graph(self, ast_root: Node) -> nx.DiGraph:
        """Build Program Dependence Graph - simplified implementation"""
        # For now, return AST graph as PDG approximation
        return self._build_ast_graph(ast_root)

    def _build_hybrid_graph(self, ast_root: Node) -> nx.DiGraph:
        """Build hybrid graph with multiple edge types - simplified implementation"""
        # Start with AST structure
        graph = self._build_ast_graph(ast_root)

        # Add next-token edges (sequential order)
        nodes = list(graph.nodes())
        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1], edge_type=EdgeType.NEXT_TOKEN.value)

        return graph

    def get_graph_statistics(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Get statistics about the graph"""
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
        }

        # Edge type distribution
        edge_types = {}
        for _, _, data in graph.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        stats["edge_type_distribution"] = edge_types

        # Node type distribution
        node_types = {}
        for _, data in graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        stats["node_type_distribution"] = node_types

        return stats

    def build_ast_edges(
        self,
        ast_root: Node,
        node_mapping: dict[Node, int],
    ) -> torch.Tensor:
        """
        Build edge index tensor for AST structure.

        Args:
            ast_root: Root node of the AST
            node_mapping: Mapping from AST nodes to indices

        Returns:
            Edge index tensor of shape [2, num_edges]
        """

        edges = []

        def traverse_for_edges(node: Node) -> None:
            if node not in node_mapping:
                return

            parent_idx = node_mapping[node]

            # Add edges to all children
            for child in node.children:
                if child in node_mapping:
                    child_idx = node_mapping[child]
                    edges.append([parent_idx, child_idx])

                # Recursively process children
                traverse_for_edges(child)

        traverse_for_edges(ast_root)

        if not edges:
            # Return empty tensor with correct shape
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t().contiguous()
