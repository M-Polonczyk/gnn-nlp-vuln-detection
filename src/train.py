#!/usr/bin/env python3
"""Emaple usage of loading and analyzing C code samples."""

import sys
from pathlib import Path

import torch
from torch_geometric.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.data_processing.dataset_loader import DiverseVulDatasetLoader
from gnn_vuln_detection.training import metrics, train_vulnerability_detector
from gnn_vuln_detection.utils import config_loader


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train, validation, and test sets."""
    import random

    # Shuffle the dataset
    dataset_list = list(dataset)
    random.shuffle(dataset_list)

    total_size = len(dataset_list)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_set = dataset_list[:train_size]
    val_set = dataset_list[train_size : train_size + val_size]
    test_set = dataset_list[train_size + val_size :]

    return train_set, val_set, test_set


def load_config():
    config = config_loader.load_all_configs()
    dataset_config = config["dataset_paths"]
    training_config = config["training"]
    model_params = config["model_params"]
    return dataset_config, training_config, model_params


def main() -> None:
    """Main function to demonstrate loading and analyzing C code samples."""
    dataset_config, training_config, model_params = load_config()
    print(f"Dataset config: {dataset_config}")
    print(f"Training config: {training_config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load a diverse vulnerability dataset
    dataset_loader = DiverseVulDatasetLoader(
        dataset_config["diversevul"]["dataset_path"],
        dataset_config["diversevul"]["metadata_path"],
    )
    dataset = dataset_loader.load_dataset()
    print(f"Loaded dataset with {len(dataset)} samples")
    # Split dataset into train, validation, and test sets
    train_set, val_set, test_set = split_dataset(dataset)
    print(f"Train set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")
    # Convert dataset to PyTorch Geometric format
    train_graphs = dataset_loader.convert_to_pyg_graphs(train_set)
    val_graphs = dataset_loader.convert_to_pyg_graphs(val_set)
    test_graphs = dataset_loader.convert_to_pyg_graphs(test_set)

    # Create DataLoader objects
    train_loader = DataLoader(
        train_graphs,
        batch_size=training_config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=training_config["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=training_config["batch_size"],
        shuffle=False,
    )

    # Train a vulnerability detector
    model, best_val_acc = train_vulnerability_detector(
        train_loader=train_loader,  # Fixed: use actual DataLoader
        val_loader=val_loader,  # Fixed: use actual DataLoader
        model_config=model_params["gcn_standard"],
        num_epochs=training_config["epochs"],
        learning_rate=training_config["learning_rate"],
        device=device,
    )

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
    metrics.plot_confusion_matrix(y_true, y_pred_labels, labels=[0, 1])

    # Save the trained model
    torch.save(model.state_dict(), "vuln_detector.pth")
    print("Model saved as 'vuln_detector.pth'")
    scripted_model = torch.jit.script(model)
    scripted_model.save("vuln_detector.pt")

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
