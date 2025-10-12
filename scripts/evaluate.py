#!/usr/bin/env python
"""Evaluate a model aganist proccessed data."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from gnn_vuln_detection.data_processing.dataset_loader import DiverseVulDatasetLoader

from gnn_vuln_detection.data_processing.graph_converter import (
    create_vulnerability_dataset,
)
from gnn_vuln_detection.models.gnn import create_vulnerability_detector
from gnn_vuln_detection.utils import file_loader


def evaluate_model(model, test_loader, device="cpu"):
    """Evaluate model on test set."""
    model.eval()
    test_correct = 0
    test_total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)

            test_correct += (pred == batch.y).sum().item()
            test_total += batch.y.size(0)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    test_acc = test_correct / test_total

    # Calculate metrics
    import numpy as np

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    pred_labels = (all_probs > 0.5).astype(int)

    tp = np.sum((pred_labels == 1) & (all_labels == 1))
    fp = np.sum((pred_labels == 1) & (all_labels == 0))
    fn = np.sum((pred_labels == 0) & (all_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    return test_acc, precision, recall, f1


def main() -> None:
    """Main function to evaluate the model."""
    # Load dataset
    dataset = DiverseVulDatasetLoader().load_dataset()
    train_data, val_data, test_data = create_vulnerability_dataset(dataset)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False)

    # Get input dimension
    input_dim = train_data[0].x.shape[1]
    print(f"Input dimension: {input_dim}")

    # Load model from file
    model = file_loader.load_model(
        "gnn_vuln_detection/models/vulnerability_detector.pth",
        create_vulnerability_detector,
        input_dim=input_dim,
    )

    # Evaluate the model
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
