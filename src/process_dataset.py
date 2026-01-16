#!/usr/bin/env python3
"""Emaple usage of loading and analyzing C code samples."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skmultilearn.model_selection import iterative_train_test_split
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.code_representation.code_representation import CodeSample
from gnn_vuln_detection.code_representation.feature_extractor import CodeGraphProcessor
from gnn_vuln_detection.data_processing.graph_converter import DataclassToGraphConverter
from gnn_vuln_detection.dataset.loaders import DiverseVulDatasetLoader
from gnn_vuln_detection.utils import config_loader


def split_indices_only(
    samples: list[CodeSample], train_ratio=0.7, cwe_to_index=None, num_classes=None
):
    """Wykonuje split tylko na indeksach, oszczędzając pamięć."""
    # 1. Przygotuj macierz etykiet (y) i indeksy próbek (X)
    # Musimy przekazać etykiety jako tablicę numpy
    for i, s in enumerate(tqdm(samples, desc="Building Label Matrix")):
        label_vec = [0] * num_classes
        if s.cwe_ids:
            for cwe in s.cwe_ids:
                if cwe in cwe_to_index:
                    label_vec[cwe_to_index[cwe]] = 1
        samples[i].cwe_ids_labeled = label_vec
    labels = np.array([s.cwe_ids_labeled for s in samples])
    indices = np.arange(len(samples)).reshape(-1, 1)  # Indeksy próbek

    # 2. Pierwszy podział: Train vs (Val + Test)
    test_val_ratio = 1 - train_ratio
    X_train_idx, _, X_temp_idx, y_temp = iterative_train_test_split(
        indices, labels, test_size=test_val_ratio
    )

    # 3. Drugi podział: Val vs Test (pół na pół z reszty)
    X_val_idx, _, X_test_idx, _ = iterative_train_test_split(
        X_temp_idx,
        y_temp,
        test_size=0.5,  # 0.5 bo val i test są zazwyczaj równe (po 15%)
    )

    # Flatten indices
    return X_train_idx.flatten(), X_val_idx.flatten(), X_test_idx.flatten()


def split_multilabel_dataset(
    samples: list[CodeSample], train_ratio=0.7, val_ratio=0.15
):
    """Stratified split of multilabel dataset into train, val, and test sets."""
    # 1. Przygotuj macierz etykiet (y) i indeksy próbek (X)
    # Musimy przekazać etykiety jako tablicę numpy
    labels = np.array([s.cwe_ids_labeled for s in samples])
    indices = np.arange(len(samples)).reshape(-1, 1)  # Indeksy próbek

    # 2. Pierwszy podział: Train vs (Val + Test)
    test_val_ratio = 1 - train_ratio
    X_train_idx, y_train, X_temp_idx, y_temp = iterative_train_test_split(
        indices, labels, test_size=test_val_ratio
    )

    # 3. Drugi podział: Val vs Test (pół na pół z reszty)
    # Obliczamy ile z 'temp' ma stanowić val_ratio w skali całości
    relative_val_size = val_ratio / test_val_ratio
    X_val_idx, y_val, X_test_idx, y_test = iterative_train_test_split(
        X_temp_idx,
        y_temp,
        test_size=0.5,  # 0.5 bo val i test są zazwyczaj równe (po 15%)
    )

    # 4. Mapowanie indeksów z powrotem na obiekty CodeSample
    train_samples = [samples[i[0]] for i in X_train_idx]
    val_samples = [samples[i[0]] for i in X_val_idx]
    test_samples = [samples[i[0]] for i in X_test_idx]

    return train_samples, val_samples, test_samples


def load_config() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = config_loader.load_all_configs()
    dataset_config = config["dataset_paths"]
    training_config = config["training"]
    model_params = config["model_params"]
    return dataset_config, training_config, model_params


def main() -> None:
    """Main function to demonstrate loading and analyzing C code samples."""

    def process_partition(indices):
        partition_samples = [samples[i] for i in indices]
        for i in tqdm(
            range(len(partition_samples)), desc="Converting samples to nx graphs"
        ):
            ast_root = ast_parser.parse_code_to_ast(
                ast_parser.cleanup_code(partition_samples[i].code)
            )
            partition_samples[i].graph = converter.ast_converter.ast_to_networkx(
                ast_root
            )
        return partition_samples

    dataset_config, _, model_params = load_config()
    num_classes = model_params["gcn_multiclass"]["num_classes"]
    cwe_to_index = {
        val["cwe_id"]: val["index"] for val in model_params["vulnerabilities"]
    }

    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )
    samples = diversevul_loader.load_dataset(list(cwe_to_index.keys()))
    train_idx, val_idx, test_idx = split_indices_only(
        samples, cwe_to_index=cwe_to_index, num_classes=num_classes
    )
    # seed = 42
    # np.random.seed(seed)
    # samples = samples[
    #     : len(samples) // 96
    # ]  # Use only part of the dataset for faster training
    np.random.shuffle(samples)
    converter = DataclassToGraphConverter()
    ast_parser = converter.ast_parser

    # Step 1: Convert samples to AST
    train_samples = process_partition(train_idx)

    # Step 2: Extract features
    processor = CodeGraphProcessor(
        node_dim=model_params["gcn_multiclass"]["hidden_dim"]
    )
    processor.fit([sample.graph for sample in train_samples])

    # Step 3: Convert samples to PyG Data objects
    def process_to_pyg(sample_list: list[CodeSample], desc="Converting to PyG data"):
        pyg_data_list = []
        for s in tqdm(sample_list, desc=desc):
            features = processor.process(s.graph)
            x = torch.tensor(features.node_features, dtype=torch.float)
            edge_index = torch.tensor(features.edge_index, dtype=torch.long)
            y = torch.tensor(s.cwe_ids_labeled, dtype=torch.float32).unsqueeze(0)
            data_dict = {
                "x": x,
                "y": y,
                "edge_index": edge_index,
                "edge_features": torch.tensor(features.edge_features, dtype=torch.float)
                if features.edge_features is not None
                else None,
            }
            pyg_data_list.append(Data(**data_dict))
        return pyg_data_list

    torch.save(
        process_to_pyg(train_samples, desc="Processing train samples"),
        "data/processed/train-diversevul-c.pt",
    )
    del train_samples  # Free memory

    test_samples = process_partition(test_idx)
    torch.save(
        process_to_pyg(test_samples, desc="Processing test samples"),
        "data/processed/test-diversevul-c.pt",
    )
    del test_samples  # Free memory

    val_samples = process_partition(val_idx)
    del samples  # Free memory
    torch.save(
        process_to_pyg(val_samples, desc="Processing val samples"),
        "data/processed/val-diversevul-c.pt",
    )
    del val_samples  # Free memory


if __name__ == "__main__":
    main()
