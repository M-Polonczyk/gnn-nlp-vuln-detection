#!/usr/bin/env python3
import gc
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skmultilearn.model_selection import iterative_train_test_split
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
# Zakładam, że te importy działają w Twoim środowisku
from gnn_vuln_detection.code_representation.code_representation import CodeSample
from gnn_vuln_detection.code_representation.feature_extractor import CodeGraphProcessor
from gnn_vuln_detection.data_processing.graph_converter import DataclassToGraphConverter
from gnn_vuln_detection.dataset.loaders import DiverseVulDatasetLoader
from gnn_vuln_detection.utils import config_loader

# --- Globalne instancje dla workerów (aby uniknąć kopiowania przy fork) ---
converter_instance = None
processor_instance = None
cwe_to_index_instance = None
num_classes_instance = None


def init_worker(processor, cwe_to_index, num_classes):
    """Inicjalizacja workera - tworzy kopie konwerterów raz na proces."""
    global \
        converter_instance, \
        processor_instance, \
        cwe_to_index_instance, \
        num_classes_instance
    converter_instance = DataclassToGraphConverter()
    processor_instance = processor
    cwe_to_index_instance = cwe_to_index
    num_classes_instance = num_classes


def load_config() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = config_loader.load_all_configs()
    dataset_config = config["dataset_paths"]
    training_config = config["training"]
    model_params = config["model_params"]
    return dataset_config, training_config, model_params


def process_single_sample(sample: CodeSample) -> Data | None:
    """
    Funkcja wykonywana równolegle.
    Zamienia CodeSample -> AST -> NetworkX -> Features -> PyG Data.
    Zwraca Data (gotowy tensor) i nie trzyma NetworkX w pamięci.
    """
    global \
        converter_instance, \
        processor_instance, \
        cwe_to_index_instance, \
        num_classes_instance

    try:
        # 1. Label Encoding
        label_vec = [0.0] * num_classes_instance
        if sample.cwe_ids:
            for cwe in sample.cwe_ids:
                if cwe in cwe_to_index_instance:
                    label_vec[cwe_to_index_instance[cwe]] = 1.0

        # 2. Parse & Graph Conversion
        ast_parser = converter_instance.ast_parser
        clean_code = ast_parser.cleanup_code(sample.code)

        # Szybki fail dla pustego kodu
        if not clean_code.strip():
            return None

        ast_root = ast_parser.parse_code_to_ast(clean_code)
        nx_graph = converter_instance.ast_converter.ast_to_networkx(ast_root)

        # 3. Feature Extraction
        # Uwaga: processor musi być już "fitted" (mieć zbudowany słownik)
        features = processor_instance.process(nx_graph)

        # 4. PyG Data Creation
        x = torch.tensor(features.node_features, dtype=torch.float)
        edge_index = torch.tensor(features.edge_index, dtype=torch.long)
        y = torch.tensor(label_vec, dtype=torch.float32).unsqueeze(0)

        edge_attr = None
        if features.edge_features is not None:
            edge_attr = torch.tensor(features.edge_features, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Opcjonalnie: dodaj metadata, jeśli potrzebujesz śledzić pliki
        # data.func_id = sample.func_id

        return data

    except Exception:
        # Ignorujemy błędy parsowania pojedynczych plików, żeby nie wywalić całego procesu
        return None


def get_graph_for_fitting(sample: CodeSample):
    """Lżejsza wersja funkcji tylko do wygenerowania grafu dla fitowania processora."""
    converter = DataclassToGraphConverter()
    try:
        code = converter.ast_parser.cleanup_code(sample.code)
        ast = converter.ast_parser.parse_code_to_ast(code)
        return converter.ast_converter.ast_to_networkx(ast)
    except:
        return None


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


def save_in_chunks(data_list: list[Data], path_prefix: str, chunk_size=20000):
    """Zapisuje listę w mniejszych plikach, aby nie zapchać RAM przy torch.save."""
    Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
    total = len(data_list)
    for i in range(0, total, chunk_size):
        chunk = data_list[i : i + chunk_size]
        file_path = f"{path_prefix}_part{i // chunk_size}.pt"
        torch.save(chunk, file_path)
        print(f"Saved {file_path}")


def main() -> None:
    dataset_config, _, model_params = load_config()
    num_classes = model_params["gcn_multiclass"]["num_classes"]
    cwe_to_index = {
        val["cwe_id"]: val["index"] for val in model_params["vulnerabilities"]
    }
    # 1. Load Raw Data (Text only)
    print("Loading dataset...")
    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )
    samples = diversevul_loader.load_dataset(list(cwe_to_index.keys()))

    # 2. Split Indices (Bez przetwarzania grafów)
    train_idx, val_idx, test_idx = split_indices_only(
        samples, cwe_to_index=cwe_to_index, num_classes=num_classes
    )

    print(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
    )

    # 3. Fit Processor (Na małym podzbiorze!)
    print("Fitting processor on random subset...")
    subset_size = min(5000, len(train_idx))
    subset_indices = np.random.choice(train_idx, subset_size, replace=False)
    subset_samples = [samples[i] for i in subset_indices]

    # Używamy prostego map, bo to mały zbiór
    graphs_for_fit = []
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(get_graph_for_fitting, subset_samples)
        graphs_for_fit = [g for g in results if g is not None]

    processor = CodeGraphProcessor(
        node_dim=model_params["gcn_multiclass"]["hidden_dim"]
    )
    processor.fit(graphs_for_fit)

    del graphs_for_fit
    del subset_samples
    gc.collect()

    # 4. Process and Save in Parallel
    # Funkcja pomocnicza do obsługi przetwarzania partycji
    def process_partition(indices, name):
        print(f"Processing {name} set ({len(indices)} samples)...")
        # Wybieramy próbki dla tej partycji
        partition_samples = [samples[i] for i in indices]

        valid_data_list = []
        # Używamy pool z initializerem, aby przekazać 'frozen' processor
        # Dajemy chunksize, aby procesy nie komunikowały się zbyt często
        with mp.Pool(
            processes=mp.cpu_count(),
            initializer=init_worker,
            initargs=(processor, cwe_to_index, num_classes),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(process_single_sample, partition_samples, chunksize=100),
                    total=len(partition_samples),
                    desc=f"Converting {name}",
                )
            )

            # Filtrujemy None (błędy parsowania)
            valid_data_list = [r for r in results if r is not None]

        print(f"Saving {name} ({len(valid_data_list)} valid samples)...")
        # Zapisujemy w jednym pliku lub w chunkach, zależnie od preferencji.
        # Przy 200k samplach, 'train' będzie miał ~140k. Lista 140k obiektów Data może ważyć 1-2GB w RAM.
        # Bezpieczniej zapisać od razu.
        save_in_chunks(valid_data_list, f"data/processed/{name}-diversevul-c")

        # Czyścimy pamięć
        del valid_data_list
        del partition_samples
        gc.collect()

    # Przetwarzamy po kolei, aby oszczędzać RAM
    process_partition(train_idx, "train")
    process_partition(val_idx, "val")
    process_partition(test_idx, "test")

    print("Done.")


if __name__ == "__main__":
    # Wymagane dla multiprocessing w niektórych środowiskach
    mp.set_start_method("spawn", force=True)
    main()
