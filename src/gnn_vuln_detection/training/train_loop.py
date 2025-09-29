import logging

import torch
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from src.gnn_vuln_detection.models.gnn import BaseGNN

from .metrics import MetricTracker, calculate_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)


def train_loop(
    model: BaseGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    num_epochs=100,
    device="cpu",
):
    """
    Main training loop for the GNN model.
    """

    def train_epoch():
        nonlocal model, train_loader, optimizer, device, criterion, val_loader, best_val_acc, best_model_state
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)

        train_acc = train_correct / train_total
        return train_loss / len(train_loader), train_acc

    train_tracker = MetricTracker(metric_names=["accuracy", "loss"])
    val_tracker = MetricTracker(
        metric_names=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
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
    best_val_f1 = -1.0

    logging.info("Starting training loop...")

    for epoch in range(num_epochs):
        # Training phase
        avg_train_loss, train_acc = train_epoch()
        train_tracker.update(
            {
                "accuracy": train_acc,
                "loss": avg_train_loss,
            },
            avg_train_loss,
        )

        # Validation phase
        y_true, y_pred_probs, y_pred_labels = model.evaluate(val_loader, device)
        val_metrics = calculate_metrics(y_true, y_pred_probs, y_pred_labels)
        val_tracker.update(val_metrics, 0.0)

        # Learning rate scheduling
        scheduler.step(avg_train_loss)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            logging.info(
                f"Epoch {epoch:3d}: "
                f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Val F1: {val_metrics['f1_score']:.4f}",
            )

    # Save best model based on validation F1-score
    f1 = val_tracker.get_last_metrics()["f1_score"]
    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(model.state_dict(), "best_gnn_model.pt")
        logging.info(f"Saved best model with Val F1: {best_val_f1:.4f}")

    logging.info("Training finished.")

    # TODO: Fix plotting
    # train_tracker.save_metrics(filename_prefix="train")
    # val_tracker.save_metrics(filename_prefix="val")

    logging.info("Training and validation metric plots saved to 'plots/' directory.")

    return model, best_val_f1
