#!/usr/bin/env python3
"""Emaple usage of loading and analyzing C code samples."""

import sys
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.data_processing.graph_converter import DataclassToGraphConverter
from gnn_vuln_detection.training import metrics, train_cwe_classifier
from gnn_vuln_detection.utils import config_loader
from src.gnn_vuln_detection.dataset.loaders import DiverseVulDatasetLoader


def split_dataset(
    dataset: list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
) -> tuple[list, list, list]:
    """Split dataset into train, validation, and test sets."""
    import random  # noqa: PLC0415

    # Shuffle the dataset
    random.shuffle(dataset)

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

    # diversevul_loader = DiverseVulDatasetLoader(
    #     dataset_path=dataset_config["diversevul"]["dataset_path"],
    # )

    # converter = DataclassToGraphConverter()
    # samples = diversevul_loader.load_dataset(list(cwe_to_index.keys()))
    # samples = samples[
    #     : len(samples)
    # ]  # Use only part of the dataset for faster training
    # dataset = create_cwe_dataset(samples=samples)
    # dataset = VulnerabilityDataset(samples=samples)
    # data = []
    # for sample in tqdm(samples, desc="Converting samples to graphs"):
    #     # initialize vector of zeros length num_classes
    #     label_vec = [0] * num_classes
    #     if sample.cwe_ids:
    #         for cwe in sample.cwe_ids:
    #             if cwe in cwe_to_index:
    #                 label_vec[cwe_to_index[cwe]] = 1
    #     sample.cwe_ids_labeled = label_vec
    #     data.append(converter.code_sample_to_pyg_data(sample))

    # torch.save(data, "data/processed/dataset-diversevul-c.pt")
    data = torch.load("data/processed/dataset-diversevul-c.pt", weights_only=False)
    train, val, test = split_dataset(data)
    # train, val, test = dataset.split()

    # Create DataLoader objects
    train_loader = DataLoader(
        train,
        batch_size=training_config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val,
        batch_size=training_config["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        test,
        batch_size=training_config["batch_size"],
        shuffle=False,
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

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_true, y_pred_probs, y_pred_labels = model.evaluate(test_loader, device)
    calculated_metrics = metrics.calculate_metrics(y_true, y_pred_probs, y_pred_labels)
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
