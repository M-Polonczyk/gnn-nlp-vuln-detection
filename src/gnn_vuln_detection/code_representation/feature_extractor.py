"""Extract features from nodes/edges"""

import re
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass
class ASTNodeFeature:
    """Encapsulates extracted features from a single AST node"""

    node_type: str
    token_value: str
    semantic_type: str
    depth: int
    children_count: int
    node_id: int
    parent_id: int
    sibling_position: int


@dataclass
class GraphFeatures:
    """Kontener na dane gotowe do zasilenia modelu GNN."""

    node_features: np.ndarray  # Shape: [num_nodes, node_dim]
    edge_index: np.ndarray  # Shape: [2, num_edges]
    edge_features: np.ndarray | None = None  # Shape: [num_edges, edge_dim]
    node_types: list[str] | None = None
    edge_types: list[str] | None = None
    global_features: np.ndarray | None = None


class CodeGraphProcessor:
    """Główny procesor integrujący ekstrakcję cech z grafów NetworkX."""

    def __init__(self, node_dim: int = 64):
        self.node_extractor = NodeFeatureExtractor(vector_dim=node_dim)
        self.edge_extractor = EdgeFeatureExtractor()
        self.graph_extractor = GraphGlobalExtractor()
        self.node_dim = node_dim

    def fit(self, graphs: list[nx.DiGraph]):
        """Buduje słowniki na podstawie zbioru treningowego."""
        self.node_extractor.build_vocabulary(graphs)
        self.edge_extractor.build_edge_vocabulary(graphs)

    def process(self, graph: nx.DiGraph) -> GraphFeatures:
        """Przekształca pojedynczy graf NetworkX w GraphFeatures."""
        # 1. Mapowanie ID węzłów na indeksy 0...N-1
        node_to_idx = {node_id: i for i, node_id in enumerate(graph.nodes())}
        num_nodes = graph.number_of_nodes()

        # 2. Ekstrakcja cech węzłów
        node_feats = np.zeros((num_nodes, self.node_dim), dtype=np.float32)
        node_types_list = []

        for node_id, data in graph.nodes(data=True):
            idx = node_to_idx[node_id]
            node_feats[idx] = self.node_extractor.extract_to_vector(data)
            node_types_list.append(data.get("node_type", "unknown"))

        # 3. Ekstrakcja krawędzi (Edge Index & Features)
        edges = list(graph.edges(data=True))
        num_edges = len(edges)
        edge_index = np.zeros((2, num_edges), dtype=np.int64)
        edge_feats = []
        edge_types_list = []

        for i, (u, v, data) in enumerate(edges):
            edge_index[0, i] = node_to_idx[u]
            edge_index[1, i] = node_to_idx[v]

            # Cechy relacyjne między węzłami
            feat = self.edge_extractor.extract_edge_features(
                data, graph.nodes[u], graph.nodes[v]
            )
            edge_feats.append(list(feat.values()))
            edge_types_list.append(data.get("edge_type", "unknown"))

        # 4. Cechy globalne grafu
        global_feat_dict = self.graph_extractor.extract_graph_features(graph)
        global_feats = np.array(list(global_feat_dict.values()), dtype=np.float32)

        return GraphFeatures(
            node_features=node_feats,
            edge_index=edge_index,
            edge_features=np.array(edge_feats, dtype=np.float32),
            node_types=node_types_list,
            edge_types=edge_types_list,
            global_features=global_feats,
        )


class NodeFeatureExtractor:
    """Ekstrakcja cech węzłów do gęstych wektorów."""

    def __init__(self, vector_dim: int = 64):
        self.vector_dim = vector_dim
        self.node_type_to_id = {}
        self.vuln_patterns = [  # TODO: Extend this list
            r"strcpy|strcat|gets|sprintf|vsprintf",  # Buffer overflow
            r"malloc|free|realloc|calloc",  # Memory mgmt
            r"scanf|bscanf|bsscanf|fgets|read|getchar|getc|getc_unlocked",  # Input
            r"system|exec|popen",  # Command injection
            r"strncpy|strncat|snprintf|vsnprintf",  # Safer string funcs
            r"\*\s*\([^)]*\+[^)]*\)",  # Unsafe pointer arithmetic
        ]

    def build_vocabulary(self, graphs: list[nx.DiGraph]):
        types = set()
        for graph in graphs:
            for _, data in graph.nodes(data=True):
                types.add(data.get("node_type", "unknown"))
        self.node_type_to_id = {t: i for i, t in enumerate(sorted(types))}

    def extract_to_vector(self, node_data: dict[str, Any]) -> np.ndarray:
        vec = np.zeros(self.vector_dim, dtype=np.float32)

        # 1. Typ węzła (One-hot proxy)
        t_id = self.node_type_to_id.get(node_data.get("node_type", "unknown"), 0)
        if t_id < 20:
            vec[t_id] = 1.0

        # 2. Struktura (indeksy 20-25)
        vec[20] = float(node_data.get("child_count", 0)) / 10.0
        vec[21] = 1.0 if node_data.get("is_named", False) else 0.0
        vec[22] = 1.0 if node_data.get("is_leaf", False) else 0.0

        # 3. Bezpieczeństwo / Tekst (indeksy 26-35)
        text = str(node_data.get("text", "")).lower()
        vec[26] = min(len(text) / 100.0, 1.0)
        for i, pattern in enumerate(self.vuln_patterns):
            if re.search(pattern, text):
                vec[27 + i] = 1.0

        # 4. Flagi syntaktyczne (indeksy 36-40)
        nt = node_data.get("node_type", "")
        vec[36] = 1.0 if "statement" in nt else 0.0
        vec[37] = 1.0 if "expression" in nt else 0.0
        vec[38] = 1.0 if "call" in nt else 0.0
        return vec


class EdgeFeatureExtractor:
    """Przetwarza krawędzie i relacje między węzłami."""

    def __init__(self):
        self.edge_type_to_id = {}

    def build_edge_vocabulary(self, graphs: list[nx.DiGraph]):
        types = set()
        for graph in graphs:
            for _, _, data in graph.edges(data=True):
                types.add(data.get("edge_type", "ast"))
        self.edge_type_to_id = {t: i for i, t in enumerate(sorted(types))}

    def extract_edge_features(
        self, edge_data: dict, src: dict, dst: dict
    ) -> dict[str, float]:
        e_type = edge_data.get("edge_type", "ast")

        # Obliczanie dystansu w kodzie
        src_byte = src.get("start_byte", 0)
        dst_byte = dst.get("start_byte", 0)

        return {
            "type_id": float(self.edge_type_to_id.get(e_type, 0)),
            "byte_dist": float(abs(dst_byte - src_byte)) / 1000.0,
            "is_forward": 1.0 if dst_byte >= src_byte else 0.0,
            "is_ast": 1.0 if e_type == "ast" else 0.0,
            "is_cfg": 1.0 if e_type in ["control_flow", "cfg"] else 0.0,
            "is_dfg": 1.0 if e_type in ["data_flow", "dfg"] else 0.0,
        }


class GraphGlobalExtractor:
    """Ekstrakcja metryk całego grafu."""

    def extract_graph_features(self, graph: nx.DiGraph) -> dict[str, float]:
        n = graph.number_of_nodes()
        return {
            "nodes_count": float(n),
            "edges_count": float(graph.number_of_edges()),
            "density": nx.density(graph),
            "is_dag": float(nx.is_directed_acyclic_graph(graph)),
            "avg_degree": sum(dict(graph.degree()).values()) / n if n > 0 else 0,
        }
