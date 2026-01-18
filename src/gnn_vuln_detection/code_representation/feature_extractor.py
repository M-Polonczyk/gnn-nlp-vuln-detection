"""Extract features from nodes/edges"""

import re
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from .graph_builder import EdgeType


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

    def __init__(self, node_dim: int = 64, edge_dim: int = 10):
        self.node_extractor = NodeFeatureExtractor(vector_dim=node_dim)
        self.edge_extractor = EdgeFeatureExtractor(vector_dim=edge_dim)
        self.graph_extractor = GraphGlobalExtractor()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

    def fit(self, graphs: list[nx.DiGraph]):
        """Buduje słowniki na podstawie zbioru treningowego."""
        self.node_extractor.build_vocabulary(graphs)

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
        edge_feats = np.zeros((num_edges, self.edge_dim), dtype=np.float32)
        edge_types_list = []

        for i, (u, v, data) in enumerate(edges):
            edge_index[0, i] = node_to_idx[u]
            edge_index[1, i] = node_to_idx[v]

            edge_feats[i] = self.edge_extractor.extract_edge_features(
                data, graph.nodes[u], graph.nodes[v]
            )
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
    """Ekstrakcja cech węzłów (AST) do gęstych wektorów."""

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
        """Tworzy mapę najczęstszych typów węzłów AST."""
        type_counts = {}
        for graph in graphs:
            for _, data in graph.nodes(data=True):
                t = data.get("node_type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

        # Sortujemy i bierzemy top N, żeby zmieścić się w wektorze
        # Zostawiamy miejsce na inne cechy (np. 30 typów max)
        max_types = min(len(type_counts), self.vector_dim // 2)
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

        self.node_type_to_id = {
            t: i for i, (t, _) in enumerate(sorted_types[:max_types])
        }
        print(f"Vocab built: {len(self.node_type_to_id)} types tracked.")

    def extract_to_vector(self, node_data: dict[str, Any]) -> np.ndarray:
        vec = np.zeros(self.vector_dim, dtype=np.float32)

        # --- SEKCJA 1: One-Hot Node Type (0 -> N) ---
        node_type = node_data.get("node_type", "unknown")
        if node_type in self.node_type_to_id:
            idx = self.node_type_to_id[node_type]
            vec[idx] = 1.0

        # Offset dla kolejnych sekcji (żeby nie nadpisać One-Hot)
        offset = len(self.node_type_to_id) + 1

        # Zabezpieczenie przed wyjściem poza zakres wektora
        if offset + 20 > self.vector_dim:
            # Fallback jeśli vector_dim jest za mały
            offset = 0

        # --- SEKCJA 2: Struktura AST ---
        # Normalizujemy wartości, aby były w zakresie 0-1 (pomaga sieciom neuronowym)

        # Ilość dzieci
        child_count = node_data.get("children_count", node_data.get("child_count", 0))
        vec[offset + 0] = min(child_count / 10.0, 1.0)

        # Głębokość w drzewie (zakładamy max depth ~50)
        depth = node_data.get("depth", 0)
        vec[offset + 1] = min(depth / 50.0, 1.0)

        # Pozycja rodzeństwa (np. czy to 1. czy 5. argument funkcji)
        sibling_pos = node_data.get("sibling_position", 0)
        vec[offset + 2] = min(sibling_pos / 10.0, 1.0)

        # Flagi boolowskie
        vec[offset + 3] = 1.0 if node_data.get("is_leaf", False) else 0.0
        vec[offset + 4] = 1.0 if node_data.get("is_named", False) else 0.0

        # --- SEKCJA 3: Analiza Tekstu i Podatności ---
        text = str(node_data.get("token_value", node_data.get("text", ""))).lower()

        # Długość tokenu
        vec[offset + 5] = min(len(text) / 20.0, 1.0)

        # Wzorce regex (czy węzeł zawiera niebezpieczną funkcję?)
        for i, pattern in enumerate(self.vuln_patterns):
            if offset + 6 + i < self.vector_dim:
                if re.search(pattern, text):
                    vec[offset + 6 + i] = 1.0

        return vec


class EdgeFeatureExtractor:
    """Przetwarza krawędzie (w tym nowe typy: Reverse, NextToken)."""

    def __init__(self, vector_dim: int = 10):
        self.vector_dim = vector_dim

    def extract_edge_features(
        self, edge_data: dict, src: dict, dst: dict
    ) -> np.ndarray:
        vec = np.zeros(self.vector_dim, dtype=np.float32)

        # 1. Typ krawędzi (One-Hot lub Enum value)
        # Obsługujemy zarówno int (Enum) jak i str
        e_type = edge_data.get("edge_type", 1)

        # Mapowanie typów na indeksy wektora
        if e_type in (EdgeType.AST_EDGE, 1, "ast"):
            vec[0] = 1.0
        elif e_type in (EdgeType.AST_REVERSE, 2, "reverse"):
            vec[1] = 1.0
        elif e_type in (EdgeType.NEXT_TOKEN, 3, "next_token"):
            vec[2] = 1.0
        else:  # Inne (np. Data Flow)
            vec[3] = 1.0

        # 2. Dystans w kodzie (Byte Distance)
        # Pomaga modelowi zrozumieć, czy "Next Token" jest blisko, czy daleko
        src_byte = src.get("end_byte", src.get("start_byte", 0))
        dst_byte = dst.get("start_byte", 0)

        dist = abs(dst_byte - src_byte)
        # Logarytmiczne skalowanie dystansu jest lepsze dla sieci niż liniowe
        # (różnica 1 vs 100 bajtów jest ważna, 1000 vs 1100 mniej)
        vec[4] = min(np.log1p(dist) / 10.0, 1.0)

        # 3. Kierunek (czy skaczemy w przód czy w tył kodu)
        vec[5] = 1.0 if dst_byte >= src_byte else -1.0

        return vec


class GraphGlobalExtractor:
    """Ekstrakcja metryk całego grafu (pomocnicze dla klasyfikacji)."""

    def extract_graph_features(self, graph: nx.DiGraph) -> dict[str, float]:
        n = graph.number_of_nodes()
        e = graph.number_of_edges()

        return {
            "nodes_count": float(n),
            "edges_count": float(e),
            "edges_per_node": float(e) / n if n > 0 else 0.0,
            "density": float(e) / (n * (n - 1)) if n > 1 else 0.0,
        }
