"""Training module."""

import logging
from typing import Literal

import torch
from torch_geometric.loader import DataLoader

from gnn_vuln_detection.models.factory import create_vulnerability_detector
from gnn_vuln_detection.utils import config_loader

from . import losses, metrics
from .train_loop import train_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

__all__ = [
    "losses",
    "metrics",
    "train_cwe_classifier",
    "train_loop",
    "train_vulnerability_detector",
]


def train_vulnerability_detector(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    model_config: dict | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
):
    """Train a vulnerability detection model."""

    if model_config is None:
        model_config = config_loader.load_config("model_params.yaml")["gcn_standard"]

    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.shape[1]

    model = create_vulnerability_detector(input_dim=input_dim, **model_config).to(
        device,
    )

    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        device,
    )  # model, best_val_acc


def train_cwe_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    model_config: dict | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
):
    """
    Train a CWE (Common Weakness Enumeration) classification model.

    This function creates and trains a graph neural network model for classifying
    code vulnerabilities into different CWE categories. It handles the complete
    training loop including validation, learning rate scheduling, and model
    checkpointing.

    Args:
        train_loader (DataLoader): PyTorch Geometric DataLoader containing training data
        val_loader (DataLoader): PyTorch Geometric DataLoader containing validation data
        model_config (dict): Configuration dictionary for model creation, must include
                            'model_type' key and other model-specific parameters
        class_labels (list): List of class labels for CWE categories
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
        learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 0.001.
        weight_decay (float, optional): Weight decay for regularization. Defaults to 1e-4.
        device (str, optional): Device to run training on ('cpu' or 'cuda'). Defaults to "cpu".

    Returns:
        tuple: A tuple containing:
            - model: The trained model with best validation performance
            - best_val_acc (float): The best validation accuracy achieved during training

    Note:
        - Input dimension is automatically inferred from the first batch
        - Uses CrossEntropyLoss for multi-class classification
        - Implements learning rate scheduling with ReduceLROnPlateau
        - Tracks multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC
        - Saves the best model state based on validation accuracy
    """

    if model_config is None:
        model_config = config_loader.load_config("model_params.yaml")["gcn_multiclass"]

    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.shape[1]

    model = create_vulnerability_detector(input_dim=input_dim, **model_config).to(
        device,
    )

    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        device,
    )  # model, best_val_acc
