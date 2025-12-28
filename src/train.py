#!/usr/bin/env python3
"""Emaple usage of loading and analyzing C code samples."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joblib import dump, load
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.code_representation.code_representation import CodeSample
from gnn_vuln_detection.code_representation.feature_extractor import CodeGraphProcessor
from gnn_vuln_detection.data_processing.graph_converter import DataclassToGraphConverter
from gnn_vuln_detection.dataset.loaders import DiverseVulDatasetLoader
from gnn_vuln_detection.training import metrics, train_cwe_classifier
from gnn_vuln_detection.utils import config_loader

cache_path = Path("data/cache/graphs")
cache_path.mkdir(parents=True, exist_ok=True)


def find_optimal_thresholds(y_true, y_probs):
    thresholds = np.linspace(0.1, 0.9, 50)
    best_thresholds = np.full(y_true.shape[1], 0.5)

    for i in range(y_true.shape[1]):  # Dla każdego CWE
        best_f1 = 0
        for t in thresholds:
            preds = (y_probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = t
    return best_thresholds


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


def split_dataset(
    dataset: list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
) -> tuple[list, list, list]:
    """Split dataset into train, validation, and test sets."""

    # Shuffle the dataset
    # np.random.shuffle(dataset)

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_set = dataset[:train_size]
    val_set = dataset[train_size : train_size + val_size]
    test_set = dataset[train_size + val_size :]

    return train_set, val_set, test_set


def load_config() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = config_loader.load_all_configs()
    dataset_config = config["dataset_paths"]
    training_config = config["training"]
    model_params = config["model_params"]
    return dataset_config, training_config, model_params


def main() -> None:
    """Main function to demonstrate loading and analyzing C code samples."""
    dataset_config, training_config, model_params = load_config()
    num_classes = model_params["gcn_multiclass"]["num_classes"]
    # build a mapping from cwe_id -> index for convenience
    cwe_to_index = {
        val["cwe_id"]: val["index"] for val in model_params["vulnerabilities"]
    }

    print(f"Training config: {training_config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load a diverse vulnerability dataset
    # dataset = create_cwe_dataset(dataset_config)
    # dataset = dataset.shuffle()

    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )

    converter = DataclassToGraphConverter()
    ast_parser = converter.ast_parser
    samples = diversevul_loader.load_dataset(list(cwe_to_index.keys()))
    np.random.shuffle(samples)
    samples = samples[
        : len(samples) // 96
    ]  # Use only part of the dataset for faster training

    def get_graph(sample: CodeSample):
        cache_file = cache_path / f"{sample.id}.pkl"
        if cache_file.exists():
            return load(cache_file)

        ast_root = ast_parser.parse_code_to_ast(ast_parser.cleanup_code(sample.code))
        graph = converter.ast_converter.ast_to_networkx(ast_root)
        dump(graph, cache_file)
        return graph

    # Step 1: Convert samples to AST
    for sample in tqdm(samples, desc="Converting samples to nx graphs"):
        label_vec = [0] * num_classes
        if sample.cwe_ids:
            for cwe in sample.cwe_ids:
                if cwe in cwe_to_index:
                    label_vec[cwe_to_index[cwe]] = 1
        sample.cwe_ids_labeled = label_vec
        sample.graph = get_graph(sample)

    train_samples, val_samples, test_samples = split_multilabel_dataset(samples)
    # Sprawdzić podział klas w datasecie

    # Step 2: Extract features
    processor = CodeGraphProcessor(
        node_dim=model_params["gcn_multiclass"]["hidden_dim"]
    )
    processor.fit([s.graph for s in train_samples])

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

    train = process_to_pyg(train_samples, desc="Processing train samples")
    val = process_to_pyg(val_samples, desc="Processing val samples")
    test = process_to_pyg(test_samples, desc="Processing test samples")

    torch.save(train, "data/processed/train-diversevul-small-c.pt")
    torch.save(val, "data/processed/val-diversevul-small-c.pt")
    torch.save(test, "data/processed/test-diversevul-small-c.pt")

    # Create DataLoader objects
    train_loader = DataLoader(
        train,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    print(
        f"Dataset split: {len(train)} train, {len(val)} val, {len(test)} test samples",
    )

    # Train a vulnerability detector
    model, best_val_acc = train_cwe_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config["epochs"],
        model_config=model_params["gcn_multiclass"],
        learning_rate=float(training_config["learning_rate"]),
        device=device,
    )

    # Save the trained model
    torch.save(model.state_dict(), "cwe_detector.pth")
    print("Model saved as 'cwe_detector.pth'")
    # scripted_model = torch.jit.script(model)
    # scripted_model.save("cwe_detector.pt")

    # thresholds = find_optimal_thresholds()

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_true, y_pred_probs, y_pred_labels = model.evaluate(test_loader, device)
    calculated_metrics = metrics.calculate_metrics(
        y_true, y_pred_probs, y_pred_labels, "macro"
    )
    acc, prec, rec, f1, roc_auc = (
        calculated_metrics["accuracy"],
        calculated_metrics["precision"],
        calculated_metrics["recall"],
        calculated_metrics["f1_score"],
        calculated_metrics["roc_auc"],
    )

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")

    # Calculate and log additional metrics
    print("\nCalculating additional metrics on test set...")
    metrics.plot_confusion_matrix(y_true, y_pred_labels, labels=cwe_to_index.keys())

    input_dim = next(iter(train_loader)).x.shape[1]
    num_nodes = 10
    x = torch.randn(num_nodes, input_dim).to(device)
    batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
    row = torch.arange(num_nodes).repeat(num_nodes)
    col = torch.arange(num_nodes).unsqueeze(1).repeat(1, num_nodes).flatten()
    edge_index = torch.stack([row, col], dim=0).to(device)

    # For some reason it doesn't work on GPU
    model.to("cpu")
    x = x.to("cpu")
    edge_index = edge_index.to("cpu")
    batch = batch.to("cpu")

    # torch._check(edge_index.size(1) > 10, "Edge index size must be greater than 10.")
    # torch._check(batch.max().item() < 100, "Batch indices exceed expected range.")

    # onnx_program = torch.onnx.export(
    #     model,
    #     (x, edge_index, batch),
    #     dynamo=True,
    #     input_names=["x", "edge_index", "batch"],
    #     output_names=["output"],
    #     export_params=True,
    # )
    # onnx_program.save("vuln_classifier_model.onnx")

    # Run classifier on a sample batch
    sample_batch = next(iter(test_loader))
    sample_batch = sample_batch.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(sample_batch.x, sample_batch.edge_index, sample_batch.batch)
        pred = out.argmax(dim=1)
        print(f"Sample batch predictions: {pred.cpu().numpy()}")


if __name__ == "__main__":
    main()
