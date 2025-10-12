from gnn_vuln_detection.utils.config_loader import load_config

from .dataset import CodeMetadata, CodeSample, LanguageEnum, VulnerabilityDataset
from .dataset_loader import DiverseVulDatasetLoader, MegaVulDatasetLoader

__all__ = [
    "CodeMetadata",
    "CodeSample",
    "DiverseVulDatasetLoader",
    "LanguageEnum",
    "MegaVulDatasetLoader",
    "VulnerabilityDataset",
    "create_cwe_dataset",
    "create_vulnerability_dataset",
]


def create_cwe_dataset() -> VulnerabilityDataset:
    diversevul_dataset_path = load_config("dataset_paths.yaml")["diversevul"]["dataset_path"]
    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=diversevul_dataset_path,
        # metadata_path=Path("data/preprocessed/diversevul/c_cpp/2023/metadata.json"),
    )
    samples = diversevul_loader.load_dataset()

    megavul_dataset_path = load_config("dataset_paths.yaml")["megavul"]["dataset_path"]
    megavul_loader = MegaVulDatasetLoader(
        dataset_path=megavul_dataset_path,
    )
    samples += megavul_loader.load_dataset()
    dataset = VulnerabilityDataset(
        samples=samples,
        include_edge_features=True,
        cache_dir="data/cache",
    )
    return dataset


def create_vulnerability_dataset() -> VulnerabilityDataset:
    raise NotImplementedError("Vulnerability classification not implemented.")
