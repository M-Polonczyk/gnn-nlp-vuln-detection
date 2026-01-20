"""
Training script for GNN vulnerability detection models.
"""

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.data_processing.dataset_loader import (
    DiverseVulDatasetLoader,
)
from gnn_vuln_detection.models.factory import create_vulnerability_detector
from gnn_vuln_detection.utils import file_loader


def load_config():
    dataset_config = file_loader.load_yaml("config/dataset_paths.yaml")
    training_config = file_loader.load_yaml("config/config.yaml")
    model_params = file_loader.load_yaml("config/model_params.yaml")
    return dataset_config, training_config, model_params


def train_vulnerability_detector(
    train_loader,
    val_loader,
    model_config,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=1e-4,
    device="cpu",
):
    """Train a vulnerability detection model."""

    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.shape[1]

    # Create model
    model = create_vulnerability_detector(input_dim=input_dim, **model_config).to(
        device,
    )

    print(
        f"Created {model_config['model_type'].upper()} model with {sum(p.numel() for p in model.parameters()):,} parameters",
    )

    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)

                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch:3d}: "
                f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}",
            )

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    return model, best_val_acc


def split_dataset(dataset: list[Any], train_split=0.6, val_split=0.2):
    """Split dataset into train, validation, and test sets."""
    n = len(dataset)
    train_size = int(n * train_split)
    val_size = int(n * val_split)

    indices = torch.randperm(n).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # train_set = dataset[train_indices]
    # val_set = dataset[val_indices]
    # test_set = dataset[test_indices]

    train_set = [dataset[i] for i in train_indices]
    val_set = [dataset[i] for i in val_indices]
    test_set = [dataset[i] for i in test_indices]

    return train_set, val_set, test_set


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
    """Main function to demonstrate loading and analyzing C code samples."""
    dataset_config, training_config, model_params = load_config()
    print(f"Dataset config: {dataset_config}")
    print(f"Training config: {training_config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        model_config=model_params,
        num_epochs=training_config["epochs"],
        learning_rate=training_config["learning_rate"],
        device=device,
    )
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    _ = evaluate_model(model, test_loader, device)

    # Save the trained model
    torch.save(model.state_dict(), "vulnerability_detector_model.pth")
    print("Model saved as 'vulnerability_detector_model.pth'")


if __name__ == "__main__":
    main()
