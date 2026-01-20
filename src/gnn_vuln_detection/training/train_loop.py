import logging
from typing import Literal

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from src.gnn_vuln_detection.models.gnn import BaseGNN
from src.gnn_vuln_detection.utils.file_loader import save_file

from .metrics import MetricTracker, calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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


def train_loop(
    model: BaseGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    num_epochs: int = 100,
    device: Literal["cpu", "cuda"] = "cpu",
    pos_weight: torch.Tensor | None = None,
    min_delta: float = 0.01,
) -> tuple[BaseGNN, float]:
    """
    Main training loop for the GNN model.
    """

    train_tracker = MetricTracker(metric_names=["loss"])
    train_loader_len = len(train_loader)
    val_tracker = MetricTracker(
        metric_names=[
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "fbeta_score",
            "roc_auc",
            "binary_accuracy",
            "any_label_true",
        ],
    )

    if pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
    )

    best_val_f1 = -1.0
    best_val_fbeta = -1.0

    logging.info("Starting training loop...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        # zamiast accuracy, liczymy tylko loss
        # multi-label accuracy ma mały sens na treningu

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)  # [B,25]

            loss = criterion(out, batch.y.float())  # BCEWithLogitsLoss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / train_loader_len

        train_tracker.update(
            {
                # "accuracy": train_acc,
                "loss": avg_train_loss,
            },
        )

        # Validation phase
        y_true, y_pred_probs, y_pred_labels = model.evaluate(val_loader, device)

        thresholds = find_optimal_thresholds(y_true, y_pred_probs)
        if hasattr(model, "label_threshold"):
            model.label_threshold = thresholds  # pyright: ignore[reportArgumentType]
        logging.debug(
            "Optimal thresholds: %s", ", ".join(f"{t:.2f}" for t in thresholds)
        )

        # y_pred_labels = (y_pred_probs >= thresholds).astype(
        #     int
        # )  # Probably can be removed

        val_metrics = calculate_metrics(
            y_true, y_pred_probs, y_pred_labels, 1.5, "macro"
        )
        val_tracker.update(val_metrics)

        # Learning rate scheduling
        current_fbeta = val_metrics["fbeta_score"]
        scheduler.step(current_fbeta)

        # Save best model based on validation F1-score
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), "best_gnn_model.pt")
            logging.info("Saved best model with Val F1: %.4f", best_val_f1)
        if current_fbeta > best_val_fbeta:
            best_val_fbeta = current_fbeta
            torch.save(model.state_dict(), "best_gnn_model_fbeta.pt")
            logging.info("Saved best model with Val Fbeta: %.4f", best_val_fbeta)
        if current_fbeta > (best_val_fbeta + min_delta):
            best_val_fbeta = current_fbeta
            torch.save(model.state_dict(), "best_gnn_model_fbeta.pt")
            logging.info("Saved best model with Val Fbeta: %.4f", best_val_fbeta)
            logging.info("Model stopped improving at epoch %d", epoch)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            logging.info(
                "Epoch %3d: Train Loss=%.4f, Val Acc=%.4f, Val F1: %.4f, Val Fbeta: %.4f",
                epoch,
                avg_train_loss,
                val_metrics["accuracy"],
                val_metrics["f1_score"],
                val_metrics["fbeta_score"],
            )
            logging.info("Thresholds: %s", ", ".join(f"{t:.2f}" for t in thresholds))
            logging.debug("Train Metrics: %s", train_tracker.get_last_metrics())
            logging.debug("Val Metrics: %s", val_metrics)

    logging.info("Training finished.")

    save_file(
        "checkpoints/optimal_thresholds.csv",
        ",".join(str(t) for t in thresholds),
    )

    try:
        train_tracker.save_metrics(filename_prefix="train")
        val_tracker.save_metrics(filename_prefix="val")
    except Exception:
        logging.exception("Error ploting metrics")

    logging.info("Training and validation metric plots saved to 'plots/' directory.")

    return model, best_val_f1  # pyright: ignore[reportReturnType]
