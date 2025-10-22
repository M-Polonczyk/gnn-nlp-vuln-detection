import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricTracker:
    """
    A class to track and store various metrics during model training and evaluation.
    """

    def __init__(self, metric_names) -> None:
        self.metric_names = metric_names
        self.history = {name: [] for name in metric_names}
        self.history["loss"] = []

    def _make_metrics(self, save_dir="plots", filename_prefix="training"):
        """
        Plots the training/validation metrics over epochs using matplotlib and seaborn.
        Saves the plots to a specified directory.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        epochs = range(1, len(self.history["loss"]) + 1)

        # Plot Loss
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=epochs, y=self.history["loss"], marker="o")
        plt.title(f"{filename_prefix.replace('_', ' ').title()} Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_loss_curve.png"))
        plt.close()

        # Plot other metrics
        for metric_name in self.metric_names:
            if self.history[metric_name]:  # Ensure metric history is not empty
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=epochs, y=self.history[metric_name], marker="o")
                plt.title(
                    f"{filename_prefix.replace('_', ' ').title()} {metric_name.replace('_', ' ').title()} Over Epochs",
                )
                plt.xlabel("Epoch")
                plt.ylabel(metric_name.replace("_", " ").title())
                plt.grid(True)
                plt.tight_layout()

        return plt

    def update(self, metrics, loss_value) -> None:
        """Updates the stored metric history."""
        if not isinstance(metrics, dict):
            metrics = dict(zip(self.metric_names, metrics, strict=False))
        for name, value in metrics.items():
            self.history[name].append(value)
        self.history["loss"].append(loss_value)

    def get_last_metrics(self):
        """Returns the last recorded metric values and loss."""
        last_metrics = {name: self.history[name][-1] for name in self.metric_names}
        last_metrics["loss"] = self.history["loss"][-1]
        return last_metrics

    def get_history(self):
        """Returns the entire metric history."""
        return self.history

    def plot_metrics(self, save_dir="plots", filename_prefix="training") -> None:
        """
        Plots the training/validation metrics over epochs using matplotlib and seaborn.
        Saves the plots to a specified directory.
        """
        self._make_metrics(save_dir, filename_prefix)
        plt.show()

    def save_metrics(self, save_dir="plots", filename_prefix="training") -> None:
        """
        Plots the training/validation metrics over epochs using matplotlib and seaborn.
        Saves the plots to a specified directory.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        epochs = range(1, len(self.history["loss"]) + 1)

        # Plot Loss
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=epochs, y=self.history["loss"], marker="o")
        plt.title(f"{filename_prefix.replace('_', ' ').title()} Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename_prefix}_loss_curve.png"))
        plt.close()

        # Plot other metrics
        for metric_name in self.metric_names:
            if self.history[metric_name]:  # Ensure metric history is not empty
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=epochs, y=self.history[metric_name], marker="o")
                plt.title(
                    f"{filename_prefix.replace('_', ' ').title()} {metric_name.replace('_', ' ').title()} Over Epochs",
                )
                plt.xlabel("Epoch")
                plt.ylabel(metric_name.replace("_", " ").title())
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        save_dir, f"{filename_prefix}_{metric_name}_curve.png"
                    ),
                )
                plt.close()


def calculate_metrics(y_true, y_pred_probs, y_pred_labels, average="binary"):
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
        if (
            y_pred_probs is not None and len(np.unique(y_true)) == 2
        ):  # ROC AUC is typically for binary classification
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_probs)
    else:
        # If only one class is present, set precision, recall, f1_score, roc_auc to 0 or nan as appropriate
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1_score"] = 0.0
        if y_pred_probs is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = 0.0  # Or np.nan

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
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
