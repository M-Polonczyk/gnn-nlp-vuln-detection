import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.utils import from_networkx
from tree_sitter import Node

from gnn_vuln_detection.code_representation.graph_builder import GraphType


class LanguageEnum(Enum):
    C = "c"
    CPP = "cpp"
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    GO = "go"
    PHP = "php"


@dataclass
class CodeMetadata:
    """Data class representing metadata for a DiverseVul sample."""

    project: str = ""
    commit_id: str = ""
    bug_info: str | None = None
    commit_url: str | None = None
    repo_url: str | None = None
    file_path: str | None = None

@dataclass
class CodeSample:
    """Dataclass representing a code sample for vulnerability detection."""

    code: str
    label: int  # 0: safe, 1: vulnerable
    language: LanguageEnum = LanguageEnum.C
    cve_id: str | None = None
    cwe_ids: list[str] | None = None
    function_name: str | None = None
    line_numbers: tuple[int, int] | None = None
    size: int | None = None
    metadata: CodeMetadata = field(default_factory=CodeMetadata)


class VulnerabilityDataset(Dataset):
    """PyTorch Geometric Dataset for vulnerability detection."""

    def __init__(
        self,
        samples: Sequence[CodeSample | Node | nx.DiGraph],
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

    def split(self, ratios: tuple[float, float, float]) -> tuple[list[int], list[int], list[int]]:
        """Split dataset indices into train, validation, and test sets."""
        assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
        total_size = len(self)
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        train_end = int(ratios[0] * total_size)
        val_end = train_end + int(ratios[1] * total_size)

        train_indices = indices[:train_end].tolist()
        val_indices = indices[train_end:val_end].tolist()
        test_indices = indices[val_end:].tolist()

        return train_indices, val_indices, test_indices
