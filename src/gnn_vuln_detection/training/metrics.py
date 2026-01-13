from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricTracker:
    """
    A class to track and store various metrics during model training and evaluation.
    """

    def __init__(self, metric_names: list[str]) -> None:
        self.metric_names = metric_names
        self._history = {name: [] for name in metric_names}
        self._history["loss"] = []

    def update(self, metrics) -> None:
        """Updates the stored metric history."""
        if not isinstance(metrics, dict):
            metrics = dict(zip(self.metric_names, metrics, strict=False))

        for name, value in metrics.items():
            self._history[name].append(value)

    def get_last_metrics(self) -> dict[str, float]:
        """Returns the last recorded metric values and loss."""
        return {name: self._history[name][-1] for name in self.metric_names}

    def get_history(self) -> dict[str, list[float]]:
        """Returns the entire metric history."""
        return self._history

    def save_metrics(self, save_dir="plots", filename_prefix="train") -> None:
        """
        Plots the training/validation metrics over epochs using matplotlib and seaborn.
        Saves the plots to a specified directory.
        """
        save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        # Plot other metrics
        for metric_name, values in self._history.items():
            if not values:
                continue
            epochs = range(1, len(values) + 1)

            plt.figure(figsize=(10, 6))
            sns.lineplot(x=epochs, y=values, marker="o")
            plt.title(
                f"{filename_prefix.replace('_', ' ').title()} {metric_name.replace('_', ' ').title()} Over Epochs",
            )
            plt.xlabel("Epoch")
            plt.ylabel(metric_name.replace("_", " ").title())
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_dir / f"{filename_prefix}_{metric_name}_curve.png")
            plt.close()


def calculate_binary_accuracy(y_true, y_pred_labels):
    """
    Calculates binary accuracy for multi-label classification.

    A sample is considered correctly classified if:
    - It is safe (all labels 0) and predicted as safe (all labels 0), or
    - It is vulnerable (at least one label 1) and at least one correct label is predicted.

    Args:
        y_true (np.array): True labels.
        y_pred_labels (np.array): Predicted class labels.

    Returns:
        float: Binary accuracy score.
    """
    is_vuln_true = (y_true.sum(axis=1) > 0).astype(int)
    is_vuln_pred = (y_pred_labels.sum(axis=1) > 0).astype(int)

    return accuracy_score(is_vuln_true, is_vuln_pred)


def calculate_any_label_true(y_true, y_pred_labels) -> float:
    """
    - If sample is SAFE (all 0) and Pred is SAFE (all 0) -> Success (1)
    - If sample is VULN and Intersection(True, Pred) > 0 -> Success (1)
    - Otherwise -> Failure (0)
    """

    # Calculate intersection (bitwise AND) sum per row
    intersection_counts = (y_true * y_pred_labels).sum(axis=1)

    # Identify correct predictions:
    # Case A: At least one correct label found (intersection > 0)
    has_match = intersection_counts > 0

    # Case B: Correctly predicted as Safe (both true and pred have 0 labels)
    true_is_safe = y_true.sum(axis=1) == 0
    pred_is_safe = y_pred_labels.sum(axis=1) == 0
    correct_safe = true_is_safe & pred_is_safe

    # Combine cases
    success_vector = has_match | correct_safe

    return success_vector.mean()


def calculate_metrics(
    y_true,
    y_pred_probs,
    y_pred_labels,
    beta=2.0,
    average: Literal["micro", "macro", "samples", "weighted", "binary"] = "binary",
) -> dict[str, float | np.ndarray]:
    """
    Calculates a set of common classification metrics.

    Args:
        y_true (np.array): True labels.
        y_pred_probs (np.array): Predicted probabilities for the positive class.
        y_pred_labels (np.array): Predicted class labels.
        average (str): Averaging method for precision, recall, f1-score (e.g., 'binary', 'macro', 'weighted').
                       Use 'binary' for binary classification, 'macro' or 'weighted' for multi-class.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    metrics = {
        "roc_auc": 0.0,
        "accuracy": accuracy_score(y_true, y_pred_labels),
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
    }

    # Handle cases where only one class is present in y_true or y_pred_labels
    # which can cause issues with precision/recall/f1_score
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_labels)) > 1:
        metrics["precision"] = precision_score(
            y_true,
            y_pred_labels,
            average=average,
            zero_division=0,
        )
        metrics["recall"] = recall_score(
            y_true,
            y_pred_labels,
            average=average,
            zero_division=0,
        )
        metrics["f1_score"] = f1_score(
            y_true,
            y_pred_labels,
            average=average,
            zero_division=0,
        )
        metrics["fbeta_score"] = fbeta_score(
            y_true,
            y_pred_labels,
            beta=beta,
            average=average,
        )
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_probs)
        if average != "binary":
            # --- Custom Metric: Binary Accuracy (Is Vulnerable vs Is Safe) ---
            # Logic: A sample is vulnerable if it has at least one label (sum > 0).
            # We compare the "vulnerability status" of true vs pred.
            metrics["binary_accuracy"] = calculate_binary_accuracy(
                y_true, y_pred_labels
            )

            # --- Custom Metric: Any Label True (Partial Match / Soft Accuracy) ---
            # Logic:
            # - If sample is SAFE (all 0) and Pred is SAFE (all 0) -> Success (1)
            # - If sample is VULN and Intersection(True, Pred) > 0 -> Success (1)
            # - Otherwise -> Failure (0)
            metrics["any_label_true"] = calculate_any_label_true(y_true, y_pred_labels)
    else:
        # If only one class is present, set precision, recall, f1_score,
        # roc_auc to 0 or nan as appropriate
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1_score"] = 0.0
        metrics["roc_auc"] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true,
    y_pred_labels,
    labels,
    save_dir="plots",
    filename="confusion_matrix.png",
) -> None:
    """
    Plots a confusion matrix using seaborn and matplotlib.

    Args:
        y_true (np.array): True labels.
        y_pred_labels (np.array): Predicted class labels.
        labels (list): List of class names (e.g., ['not_vulnerable', 'vulnerable']).
        save_dir (str): Directory to save the plot.
        filename (str): Name of the file to save the plot.
    """
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    multiclass_count_threshold = 2
    if len(labels) < multiclass_count_threshold:
        msg = "At least two labels are required to plot a confusion matrix."
        raise ValueError(msg)
    if len(labels) > multiclass_count_threshold:
        cm = multilabel_confusion_matrix(y_true, y_pred_labels)
    else:
        cm = confusion_matrix(y_true, y_pred_labels)

    performance = []
    for i, label in enumerate(labels):
        _, fp, fn, tp = cm[i].ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        performance.append([label, tp, fp, fn, precision, recall])

    df_perf = pd.DataFrame(
        performance, columns=["CWE", "TP", "FP", "FN", "Precision", "Recall"]
    )

    plt.figure(figsize=(12, 10))
    df_plot = df_perf.set_index("CWE")[["TP", "FP", "FN"]]
    sns.heatmap(df_plot, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / filename)
    plt.close()

    df_perf.to_csv(Path(save_dir) / "cwe_metrics.csv", index=False)
