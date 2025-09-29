"""Model factory for creating vulnerability detection GNN models."""

from typing import Any

from .gnn import (
    BaseGNN,
    VulnerabilityGAT,
    VulnerabilityGCN,
)


class GNNModelFactory:
    """Factory for creating GNN models for vulnerability detection."""

    MODEL_TYPES = {
        "gcn": VulnerabilityGCN,
        "gat": VulnerabilityGAT,
    }

    DEFAULT_CONFIGS = {
        "gcn": {
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout_rate": 0.3,
            "use_batch_norm": True,
            "pool_type": "mean",
        },
        "gat": {
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout_rate": 0.3,
            "heads": 4,
            "concat_heads": True,
        },
        "graphsage": {
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout_rate": 0.3,
            "aggr": "mean",
        },
        "gin": {
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout_rate": 0.3,
            "eps": 0.1,
        },
    }

    @classmethod
    def create_model(
        cls,
        model_type: str,
        input_dim: int,
        num_classes: int = 2,
        config: dict[str, Any] | None = None,
    ) -> BaseGNN:
        """
        Create a GNN model for vulnerability detection.

        Args:
            model_type: Type of model ('gcn', 'gat', 'graphsage', 'gin')
            input_dim: Dimension of input node features
            num_classes: Number of output classes (default: 2 for binary classification)
            config: Additional configuration parameters

        Returns:
            Configured GNN model

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls.MODEL_TYPES:
            msg = (
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(cls.MODEL_TYPES.keys())}"
            )
            raise ValueError(msg)

        # Get default config and update with provided config
        model_config = cls.DEFAULT_CONFIGS[model_type].copy()
        if config:
            model_config.update(config)

        # Create model
        model_class = cls.MODEL_TYPES[model_type]
        return model_class(
            input_dim=input_dim,
            num_classes=num_classes,
            **model_config,
        )

    @classmethod
    def get_recommended_config(
        cls, dataset_size: int, complexity: str = "medium",
    ) -> dict[str, dict[str, Any]]:
        """
        Get recommended configurations based on dataset size and complexity.

        Args:
            dataset_size: Number of samples in dataset
            complexity: Complexity level ('low', 'medium', 'high')

        Returns:
            Dictionary with recommended configs for each model type
        """
        if dataset_size < 1000:
            base_hidden = 64
            base_layers = 2
        elif dataset_size < 10000:
            base_hidden = 128
            base_layers = 3
        else:
            base_hidden = 256
            base_layers = 4

        complexity_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
        }

        multiplier = complexity_multipliers.get(complexity, 1.0)
        hidden_dim = int(base_hidden * multiplier)

        configs = {}
        for model_type in cls.MODEL_TYPES:
            config = cls.DEFAULT_CONFIGS[model_type].copy()
            config["hidden_dim"] = hidden_dim
            config["num_layers"] = min(base_layers, 5)  # Cap at 5 layers
            configs[model_type] = config

        return configs


def create_vulnerability_detector(
    model_type: str = "gcn",
    input_dim: int = 64,
    num_classes: int = 2,
    **kwargs,
):
    """
    Convenience function to create a vulnerability detection model.

    Args:
        model_type: Type of GNN model to use
        input_dim: Dimension of input node features
        num_classes: Number of output classes
        **kwargs: Additional model configuration

    Returns:
        Configured GNN model for vulnerability detection
    """
    return GNNModelFactory.create_model(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        config=kwargs,
    )
