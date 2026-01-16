#!/usr/bin/env python3
"""Example usage of loading and analyzing C code samples."""

import multiprocessing
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

# --- Multiprocessing Workers ---

# Global variables for workers to avoid passing large objects
_worker_converter = None
_worker_cwe_to_index = None
_worker_num_classes = None
_worker_processor = None


def _init_graph_worker(cwe_to_index: dict, num_classes: int):
    """
    Initializer for the multiprocessing worker in Step 1.
    Sets up the converter and constants.
    """
    global _worker_converter, _worker_cwe_to_index, _worker_num_classes
    _worker_converter = DataclassToGraphConverter()
    _worker_cwe_to_index = cwe_to_index
    _worker_num_classes = num_classes


def _process_graph_task(sample: CodeSample) -> CodeSample:
    """
    Worker task for converting a raw CodeSample to a graph.
    Runs inside a worker process.
    """
    label_vec = [0] * _worker_num_classes
    if sample.cwe_ids:
        for cwe in sample.cwe_ids:
            if cwe in _worker_cwe_to_index:
                label_vec[_worker_cwe_to_index[cwe]] = 1
    sample.cwe_ids_labeled = label_vec

    # Use the pre-initialized converter
    ast_parser = _worker_converter.ast_parser
    try:
        # Cleanup and parsing can fail on malformed code
        code = ast_parser.cleanup_code(sample.code)
        ast_root = ast_parser.parse_code_to_ast(code)
        sample.graph = _worker_converter.ast_converter.ast_to_networkx(ast_root)
    except Exception as e:
        # In the original flow, errors might propagate.
        # We allow it to propagate to fail fast or wrap in try/except in main if robustness needed.
        raise e
    return sample


def _init_pyg_worker(processor: CodeGraphProcessor):
    """
    Initializer for the multiprocessing worker in Step 3.
    """
    global _worker_processor
    _worker_processor = processor


def _process_pyg_task(sample: CodeSample) -> Data:
    """
    Worker task for converting a graph CodeSample to PyG Data.
    """
    features = _worker_processor.process(sample.graph)
    x = torch.tensor(features.node_features, dtype=torch.float)
    edge_index = torch.tensor(features.edge_index, dtype=torch.long)
    y = torch.tensor(sample.cwe_ids_labeled, dtype=torch.float32).unsqueeze(0)

    data_dict = {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "edge_features": torch.tensor(features.edge_features, dtype=torch.float)
        if features.edge_features is not None
        else None,
    }
    return Data(**data_dict)


# --- Main Script ---


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


def process_to_pyg_parallel(
    sample_list: list[CodeSample], processor: CodeGraphProcessor, desc: str
) -> list[Data]:
    """Helper to run PyG conversion in parallel."""
    if not sample_list:
        return []
    # Determine optimal number of workers
    num_workers = multiprocessing.cpu_count()
    chunk_size = max(1, len(sample_list) // (num_workers * 4))

    results = []
    # Must use context manager to ensure pool is closed
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_init_pyg_worker,
        initargs=(processor,),
    ) as pool:
        # Use imap for lazy iteration + tqdm
        for data in tqdm(
            pool.imap(_process_pyg_task, sample_list, chunksize=chunk_size),
            total=len(sample_list),
            desc=desc,
        ):
            results.append(data)
    return results


def main() -> None:
    """Main function to demonstrate loading and analyzing C code samples."""
    dataset_config, _, model_params = load_config()
    num_classes = model_params["gcn_multiclass"]["num_classes"]
    # build a mapping from cwe_id -> index for convenience
    cwe_to_index = {
        val["cwe_id"]: val["index"] for val in model_params["vulnerabilities"]
    }

    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )

    samples = diversevul_loader.load_dataset(list(cwe_to_index.keys()))
    # seed = 42
    # np.random.seed(seed)
    np.random.shuffle(samples)

    # Step 1: Convert samples to AST (Parallel)
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} workers for AST processing.")

    chunk_size = max(1, len(samples) // (num_workers * 4))

    processed_samples = []
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_init_graph_worker,
        initargs=(cwe_to_index, num_classes),
    ) as pool:
        for sample in tqdm(
            pool.imap(_process_graph_task, samples, chunksize=chunk_size),
            total=len(samples),
            desc="Converting samples to nx graphs (Parallel)",
        ):
            processed_samples.append(sample)

    samples = processed_samples

    train_samples, val_samples, test_samples = split_multilabel_dataset(samples)
    del samples  # Free memory

    # Step 2: Extract features
    processor = CodeGraphProcessor(
        node_dim=model_params["gcn_multiclass"]["hidden_dim"]
    )
    print("Fitting processor on train samples...")
    processor.fit([s.graph for s in train_samples])

    # Step 3: Convert samples to PyG Data objects (Parallel)
    train_data = process_to_pyg_parallel(
        train_samples, processor, desc="Processing train samples (Parallel)"
    )
    torch.save(
        train_data,
        "data/processed/train-diversevul-c.pt",
    )
    del train_data

    test_data = process_to_pyg_parallel(
        test_samples, processor, desc="Processing test samples (Parallel)"
    )
    torch.save(
        test_data,
        "data/processed/test-diversevul-c.pt",
    )
    del test_data

    val_data = process_to_pyg_parallel(
        val_samples, processor, desc="Processing val samples (Parallel)"
    )
    torch.save(
        val_data,
        "data/processed/val-diversevul-c.pt",
    )
    del val_data


if __name__ == "__main__":
    main()
