from typing import Any, Dict

from .dataset import VulnerabilityDataset
from .dataset_loader import DiverseVulDatasetLoader, MegaVulDatasetLoader

__all__ = [
    "DiverseVulDatasetLoader",
    "MegaVulDatasetLoader",
    "VulnerabilityDataset",
    "create_cwe_dataset",
    "create_vulnerability_dataset",
]


def create_cwe_dataset(
    dataset_config: dict[str, Any] | None = None,
) -> VulnerabilityDataset:
    if not dataset_config:
        from gnn_vuln_detection.utils.config_loader import load_config

        dataset_config = load_config("dataset_paths.yaml")

    diversevul_dataset_path = dataset_config["diversevul"]["dataset_path"]
    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=diversevul_dataset_path,
        # metadata_path=Path("data/preprocessed/diversevul/c_cpp/2023/metadata.json"),
    )
    samples = diversevul_loader.load_dataset()

    # megavul_dataset_path = dataset_config["megavul"]["dataset_path"]
    # megavul_loader = MegaVulDatasetLoader(
    #     dataset_path=megavul_dataset_path,
    # )
    # samples += megavul_loader.load_dataset()
    return VulnerabilityDataset(
        root=dataset_config.get("root_data_dir", "data"),
        samples=samples,
        include_edge_features=True,
        # cache_dir="data/cache",
    )


def create_vulnerability_dataset() -> VulnerabilityDataset:
    msg = "Vulnerability classification not implemented."
    raise NotImplementedError(msg)
