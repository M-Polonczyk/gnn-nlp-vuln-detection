"""Dataset loading utilities for GNN-based vulnerability detection"""
import abc
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset

from gnn_vuln_detection.code_representation.ast_parser import ASTParser
from gnn_vuln_detection.code_representation.feature_extractor import ASTFeatureExtractor
from gnn_vuln_detection.code_representation.graph_builder import GraphBuilder
from gnn_vuln_detection.utils.file_loader import load_json

logger = logging.getLogger(__name__)


def _parse_git_url(git_url: str) -> tuple[str | None, str | None]:
    """
    Parses a Git URL to extract project name and commit ID.
    Supports common GitHub and GitLab commit URL patterns.
    """
    if not git_url:
        return None, None

    # GitHub: https://github.com/owner/repo/commit/hash
    github_match = re.match(
        r"https://github\.com/([^/]+)/([^/]+)/commit/([0-9a-fA-F]{7,})",
        git_url,
    )
    if github_match:
        owner, repo, commit_hash = github_match.groups()
        project_name = f"{owner}/{repo}"
        return project_name, commit_hash

    # GitLab: https://gitlab.com/namespace/project/-/commit/hash
    # The project path can include subgroups: namespace/subgroup/project
    gitlab_match = re.match(
        r"https://gitlab\.com/((?:[^/]+/)*[^/]+)/-/commit/([0-9a-fA-F]{7,})",
        git_url,
    )
    if gitlab_match:
        project_full_path, commit_hash = gitlab_match.groups()
        return project_full_path, commit_hash

    logger.warning(
        f"Could not parse project/commit_id from git_url: {git_url} using common patterns.",
    )
    return None, None


def save_json(data: list | dict, file_path: Path) -> int:
    """Save data to a JSON file. Returns 1 on success, 0 on failure."""
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logging.exception("Could not write to JSON file: %s - %s", file_path, e)
        return 0
    return 1


@dataclass
class DiverseVulSample:
    """Data class representing a single DiverseVul dataset sample."""

    func: str  # Function source code
    target: int  # 0 for non-vulnerable, 1 for vulnerable
    cwe: list[str]  # List of CWE identifiers
    project: str  # Project name
    commit_id: str  # Git commit ID
    hash_value: int  # Hash value of the function
    size: int  # Size of the function
    message: str  # Commit message

    def __post_init__(self):
        """Validate the sample data after initialization."""
        if self.target not in [0, 1]:
            msg = f"Target must be 0 or 1, got {self.target}"
            raise ValueError(msg)
        if not isinstance(self.cwe, list):
            self.cwe = []
        if not isinstance(self.func, str) or not self.func.strip():
            msg = "Function code cannot be empty"
            raise ValueError(msg)


@dataclass
class DiverseVulMetadata:
    """Data class representing metadata for a DiverseVul sample."""

    project: str
    commit_id: str
    cwe: str | None = None
    cve: str | None = None
    bug_info: str | None = None
    commit_url: str | None = None
    repo_url: str | None = None


@dataclass
class MegaVulSample:
    """Data class representing a single MegaVul dataset sample based on megavul_simple.json schema."""

    # Fields directly from or corresponding to megavul_simple.json item
    is_vul: bool
    func_after: str  # Corresponds to item['func'] (fixed function)
    git_url: str | None = None  # item['git_url']

    cve_id: str | None = None  # item.get('cve_id')
    cvss_vector: str | None = None  # item.get('cvss_vector')
    func_before: str | None = (
        None  # item.get('func_before'), relevant if is_vul is True
    )
    abstract_func_after: str | None = None  # item.get('abstract_func')
    # item.get('diff_line_info'), defaults to empty dict if missing/None in JSON
    diff_line_info: dict[str, list[str]] = field(default_factory=dict)

    func_graph_path_before: str | None = None  # item.get('func_graph_path_before')
    func_graph_path_after: str | None = (
        None  # Assumed: item.get('func_graph_path_after')
    )

    # Derived fields, populated by the loader
    project: str | None = None
    commit_id: str | None = None

    def __post_init__(self):
        """Validate the sample data after initialization."""
        if self.is_vul and self.func_before is None:
            logger.debug(
                f"Vulnerable sample (git_url: {self.git_url}, cve_id: {self.cve_id}) has no func_before.",
            )
        if not self.is_vul and self.func_before is not None:
            logger.debug(
                f"Non-vulnerable sample (git_url: {self.git_url}, cve_id: {self.cve_id}) has func_before content.",
            )
        if self.git_url and (self.project is None or self.commit_id is None):
            logger.debug(
                f"Sample with git_url {self.git_url} is missing derived project/commit_id.",
            )
        if not self.func_after or not self.func_after.strip():
            msg = f"func_after (fixed function code) cannot be empty. Sample: git_url: {self.git_url}, cve_id: {self.cve_id}"
            raise ValueError(
                msg,
            )


class CodeGraphDataset(Dataset):
    def __init__(self, raw_code_samples, labels, transform=None, pre_transform=None) -> None:
        super().__init__(transform, pre_transform)
        # Tutaj logika ładowania danych, np. z plików lub list
        # i konwersji do listy obiektów Data
        self.data_list = []
        for _code, _label in zip(raw_code_samples, labels, strict=False):
            # graph_data = code_to_graph(code, label) # Użyj swojej funkcji
            # self.data_list.append(graph_data)
            pass  # Placeholder

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class DatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self.samples = []
        if not self.dataset_path.exists():
            msg = f"Dataset file not found at {self.dataset_path}"
            raise FileNotFoundError(msg)

    @abc.abstractmethod
    def _serialize_data(self, samples) -> list[dict]:
        """Serialize the dataset samples to a format suitable for saving."""

    @abc.abstractmethod
    def load_dataset(self) -> list[dict]:
        """Load the dataset from the JSON file."""

    def save_processed_dataset(
        self,
        output_path: str | Path,
        samples_to_save: list[MegaVulSample] | None = None,
    ) -> None:
        """
        Save processed dataset to JSON file, attempting to match megavul_simple.json key names.

        Args:
            output_path: Path to save the processed dataset.
            samples_to_save: Optional list of samples to save (defaults to all loaded samples).
        """
        output_path = Path(output_path)
        active_samples = (
            samples_to_save if samples_to_save is not None else self.samples
        )

        if not active_samples:
            logger.warning("No samples to save.")
            return

        serializable_data = self._serialize_data(active_samples)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_json(serializable_data, output_path)

    def _validate_pyg_data(self, data: Data) -> bool:
        """Validate PyTorch Geometric Data object."""
        try:
            # Check if data has required attributes
            if (
                not hasattr(data, "x")
                or not hasattr(data, "edge_index")
                or not hasattr(data, "y")
            ):
                return False

            # Check tensor shapes
            if data.x.shape[0] == 0:
                return False

            if data.edge_index.shape[0] != 2:
                # Wrong edge format
                return False

            # Check edge indices are within bounds
            if data.edge_index.shape[1] > 0:
                # There are edges
                max_edge_idx = data.edge_index.max().item()
                if max_edge_idx >= data.x.shape[0]:
                    # Edge index out of bounds
                    return False

            return True
        except Exception:
            return False


class MegaVulDatasetLoader(DatasetLoader):
    """Loader for MegaVul vulnerability dataset (megavul_simple.json format)."""

    def __init__(
        self,
        dataset_path: str | Path,
        graph_dir_path: str | Path,
    ) -> None:
        """
        Initialize the MegaVul dataset loader.

        Args:
            dataset_path: Path to the main dataset JSON file (e.g., megavul_simple.json).
            graph_dir_path: Path to the directory containing graph JSON files.
        """
        super().__init__(dataset_path)
        self.graph_dir_path = Path(graph_dir_path)

        if not self.graph_dir_path.is_dir():  # Check if it's a directory
            msg = f"Graph directory not found or is not a directory at {self.graph_dir_path}"
            raise FileNotFoundError(
                msg,
            )

    def _serialize_data(self, samples: list[MegaVulSample]) -> list[dict]:
        """
        Serialize the dataset samples to a format suitable for saving.

        Args:
            samples: List of MegaVulSample objects to serialize.

        Returns:
            List of dictionaries representing the serialized samples.
        """
        serializable_data = []
        for sample in samples:
            item = {
                "cve_id": sample.cve_id,
                "cvss_vector": sample.cvss_vector,
                "is_vul": sample.is_vul,
                "func_before": sample.func_before,
                "func": sample.func_after,  # 'func' key in JSON is the fixed function
                "abstract_func": sample.abstract_func_after,
                "diff_line_info": (
                    sample.diff_line_info if sample.diff_line_info else None
                ),
                "git_url": sample.git_url,
                "func_graph_path_before": sample.func_graph_path_before,
                "func_graph_path_after": sample.func_graph_path_after,
            }
            # Remove keys with None values for cleaner output, common in these types of JSON files.
            serializable_data.append({k: v for k, v in item.items() if v is not None})

        return serializable_data

    def load_dataset(self) -> list[MegaVulSample]:
        """
        Load the main dataset from JSON file.

        Returns:
            List of MegaVulSample objects
        """
        logger.info(f"Loading dataset from {self.dataset_path}")

        raw_data = load_json(self.dataset_path)

        self.samples = []
        skipped_samples_count = 0

        for i, item in enumerate(raw_data):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dictionary item at index {i} in dataset.")
                skipped_samples_count += 1
                continue
            try:
                git_url = item.get("git_url")
                project, commit_id = None, None
                if isinstance(git_url, str):
                    project, commit_id = _parse_git_url(git_url)
                elif git_url is not None:
                    logger.warning(
                        f"git_url for item {i} is not a string: {git_url}. Cannot parse project/commit.",
                    )

                # func_before should only be populated if is_vul is True and func_before exists
                is_vul_val = item["is_vul"]
                func_before_val = None
                if is_vul_val:
                    func_before_val = item.get("func_before")

                sample = MegaVulSample(
                    is_vul=is_vul_val,
                    func_after=item["func"],  # 'func' key in JSON is the fixed function
                    git_url=(git_url if isinstance(git_url, str) else None),
                    cve_id=item.get("cve_id"),
                    cvss_vector=item.get("cvss_vector"),
                    func_before=func_before_val,
                    abstract_func_after=item.get("abstract_func"),
                    diff_line_info=item.get("diff_line_info", {}),
                    func_graph_path_before=item.get("func_graph_path_before"),
                    func_graph_path_after=item.get("func_graph_path_after"),
                    project=project,
                    commit_id=commit_id,
                )
                self.samples.append(sample)

            except (KeyError, ValueError) as e:
                logger.warning(
                    f"Skipping sample at index {i} due to error: {e}. Item snapshot: {str(item)[:200]}gnn_vuln_detection..",
                )
                skipped_samples_count += 1
                continue
            except Exception as e:
                logger.error(
                    f"Unexpected error processing sample at index {i}: {e}. Item snapshot: {str(item)[:200]}gnn_vuln_detection..",
                    exc_info=True,
                )
                skipped_samples_count += 1
                continue

        logger.info(
            f"Successfully loaded {len(self.samples)} samples. Skipped {skipped_samples_count} samples.",
        )
        return self.samples

    def load_graph_data(
        self,
        sample: MegaVulSample,
        graph_type: str = "before",
    ) -> dict | None:
        """
        Load graph data (nodes, edges) for a given sample.

        Args:
            sample: The MegaVulSample object.
            graph_type: "before" for func_before graph, "after" for func_after graph.

        Returns:
            Dictionary containing 'nodes' and 'edges', or None if graph not found/loadable.
        """
        graph_path_str: str | None = None
        if graph_type == "before":
            graph_path_str = sample.func_graph_path_before
        elif graph_type == "after":
            graph_path_str = sample.func_graph_path_after
        else:
            logger.error(
                f"Invalid graph_type: '{graph_type}'. Must be 'before' or 'after'.",
            )
            return None

        if not graph_path_str:
            # This is not an error, just means no graph path is available for this type.
            # logger.debug(f"No graph path specified for sample (CVE: {sample.cve_id}, Git: {sample.git_url}, Type: {graph_type}).")
            return None

        graph_file_path = self.graph_dir_path / graph_path_str

        if not graph_file_path.exists():
            logger.warning(
                f"Graph file not found: {graph_file_path} (for sample CVE: {sample.cve_id}, Git: {sample.git_url}, Type: {graph_type}).",
            )
            return None

        try:
            with graph_file_path.open("r", encoding="utf-8") as f:
                graph_data = json.load(f)
            if (
                not isinstance(graph_data, dict)
                or "nodes" not in graph_data
                or "edges" not in graph_data
            ):
                logger.warning(
                    f"Graph file {graph_file_path} is malformed or missing 'nodes'/'edges' keys.",
                )
                return None
            return graph_data
        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON in graph file {graph_file_path}: {e}")
            return None
        except Exception as e:  # Catch any other OS or permission errors
            logger.error(
                f"Error loading graph file {graph_file_path}: {e}",
                exc_info=True,
            )
            return None

    def filter_by_is_vul(self, is_vul: bool) -> list[MegaVulSample]:
        """Filter samples by vulnerability status."""
        return [sample for sample in self.samples if sample.is_vul == is_vul]

    def filter_by_cve_id(self, cve_ids: list[str]) -> list[MegaVulSample]:
        """Filter samples by a list of CVE identifiers."""
        target_cve_ids = {cve_id for cve_id in cve_ids if cve_id}
        return [
            sample
            for sample in self.samples
            if sample.cve_id and sample.cve_id in target_cve_ids
        ]

    def filter_by_project(self, projects: list[str]) -> list[MegaVulSample]:
        """Filter samples by a list of project names (e.g., ["owner/repo"])."""
        target_projects = {p for p in projects if p}  # Ensure no None/empty strings
        return [
            sample
            for sample in self.samples
            if sample.project and sample.project in target_projects
        ]

    def get_statistics(self) -> dict:
        """Get various statistics about the loaded dataset."""
        if not self.samples:
            return {
                "total_samples": 0,
                "vulnerable_samples": 0,
                "non_vulnerable_samples": 0,
                "vulnerability_ratio": 0,
                "unique_cve_ids": 0,
                "cve_id_list": [],
                "unique_projects": 0,
                "project_list": [],
                "samples_with_func_before": 0,
                "samples_with_abstract_func_after": 0,
                "samples_with_graph_before": 0,
                "samples_with_graph_after": 0,
            }

        vulnerable_count = sum(1 for s in self.samples if s.is_vul)
        non_vulnerable_count = len(self.samples) - vulnerable_count

        all_cve_ids = sorted({s.cve_id for s in self.samples if s.cve_id})
        all_projects = sorted({s.project for s in self.samples if s.project})

        return {
            "total_samples": len(self.samples),
            "vulnerable_samples": vulnerable_count,
            "non_vulnerable_samples": non_vulnerable_count,
            "vulnerability_ratio": (
                vulnerable_count / len(self.samples) if self.samples else 0
            ),
            "unique_cve_ids": len(all_cve_ids),
            "cve_id_list": all_cve_ids,
            "unique_projects": len(all_projects),
            "project_list": all_projects,
            "samples_with_func_before": sum(1 for s in self.samples if s.func_before),
            "samples_with_abstract_func_after": sum(
                1 for s in self.samples if s.abstract_func_after
            ),
            "samples_with_graph_before": sum(
                1 for s in self.samples if s.func_graph_path_before
            ),
            "samples_with_graph_after": sum(
                1 for s in self.samples if s.func_graph_path_after
            ),
        }

    # def save_processed_dataset(
    #     self,
    #     output_path: Union[str, Path],
    #     samples_to_save: Optional[List[MegaVulSample]] = None,
    # ):
    #     """
    #     Save processed dataset to JSON file, attempting to match megavul_simple.json key names.

    #     Args:
    #         output_path: Path to save the processed dataset.
    #         samples_to_save: Optional list of samples to save (defaults to all loaded samples).
    #     """
    #     output_path = Path(output_path)
    #     active_samples = (
    #         samples_to_save if samples_to_save is not None else self.samples
    #     )

    #     if not active_samples:
    #         logger.warning("No samples to save.")
    #         return

    #     serializable_data = []
    #     for sample in active_samples:
    #         item = {
    #             "cve_id": sample.cve_id,
    #             "cvss_vector": sample.cvss_vector,
    #             "is_vul": sample.is_vul,
    #             "func_before": sample.func_before,
    #             "func": sample.func_after,  # Map back to 'func' key
    #             "abstract_func": sample.abstract_func_after,  # Map back to 'abstract_func'
    #             "diff_line_info": (
    #                 sample.diff_line_info if sample.diff_line_info else None
    #             ),  # Use None if empty dict for cleaner JSON
    #             "git_url": sample.git_url,
    #             "func_graph_path_before": sample.func_graph_path_before,
    #             "func_graph_path_after": sample.func_graph_path_after,
    #             # Derived fields (project, commit_id) are typically not saved if git_url is present,
    #             # as they can be re-derived.
    #         }
    #         # Remove keys with None values for cleaner output, common in these types of JSON files.
    #         serializable_data.append({k: v for k, v in item.items() if v is not None})

    #     output_path.parent.mkdir(parents=True, exist_ok=True)

    #     save_json(serializable_data, output_path)


class DiverseVulDatasetLoader(DatasetLoader):
    """Loader for DiverseVul vulnerability dataset."""

    def __init__(
        self,
        dataset_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the DiverseVul dataset loader.

        Args:
            dataset_path: Path to the main dataset JSON file
            metadata_path: Optional path to the metadata JSON file
        """
        super().__init__(dataset_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.samples: list[DiverseVulSample] = []
        self.metadata: dict[str, DiverseVulMetadata] = {}

        if self.metadata_path and not self.metadata_path.exists():
            logger.warning(f"Metadata file not found at {self.metadata_path}")
            self.metadata_path = None

    def load_dataset(self) -> list[DiverseVulSample]:
        """
        Load the main dataset from JSON file.

        Returns:
            List of DiverseVulSample objects
        """
        logger.info(f"Loading DiverseVul dataset from {self.dataset_path}")

        raw_data = load_json(self.dataset_path)

        self.samples = []
        skipped_samples = 0

        for i, item in enumerate(raw_data):
            try:
                sample = DiverseVulSample(
                    func=item["func"],
                    target=item["target"],
                    cwe=item.get("cwe", []),
                    project=item["project"],
                    commit_id=item["commit_id"],
                    hash_value=item["hash"],
                    size=item["size"],
                    message=item["message"],
                )
                self.samples.append(sample)
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping sample {i}: {e}")
                skipped_samples += 1
                continue

        logger.info(
            f"Loaded {len(self.samples)} DiverseVul samples, skipped {skipped_samples}",
        )
        return self.samples

    def load_metadata(self) -> dict[str, DiverseVulMetadata]:
        """
        Load metadata from JSON file if available.

        Returns:
            Dictionary mapping commit_id to DiverseVulMetadata
        """
        if not self.metadata_path:
            logger.info("No metadata file provided")
            return {}

        logger.info(f"Loading DiverseVul metadata from {self.metadata_path}")

        try:
            with self.metadata_path.open("r", encoding="utf-8") as f:
                raw_metadata = json.load(f)
        except json.JSONDecodeError:
            logger.exception("Invalid JSON in metadata file:")
            return {}
        except UnicodeDecodeError:
            logger.exception("Encoding error in metadata file:")
            return {}

        if not isinstance(raw_metadata, list):
            logger.error("Metadata JSON should contain a list of entries")
            return {}

        self.metadata = {}
        for item in raw_metadata:
            try:
                metadata = DiverseVulMetadata(
                    project=item["project"],
                    commit_id=item["commit_id"],
                    cwe=item.get("CWE"),
                    cve=item.get("CVE"),
                    bug_info=item.get("bug_info"),
                    commit_url=item.get("commit_url"),
                    repo_url=item.get("repo_url"),
                )
                # Use project_commit_id as key for lookup
                key = f"{metadata.project}_{metadata.commit_id}"
                self.metadata[key] = metadata
            except KeyError as e:
                logger.warning(f"Skipping metadata entry due to missing key: {e}")
                continue

        logger.info(f"Loaded metadata for {len(self.metadata)} DiverseVul entries")
        return self.metadata

    def get_sample_with_metadata(
        self,
        sample: DiverseVulSample,
    ) -> tuple[DiverseVulSample, DiverseVulMetadata | None]:
        """
        Get a sample along with its corresponding metadata.

        Args:
            sample: DiverseVulSample object

        Returns:
            Tuple of (sample, metadata) where metadata can be None if not found
        """
        key = f"{sample.project}_{sample.commit_id}"
        metadata = self.metadata.get(key)
        return sample, metadata

    def filter_by_target(self, target: int) -> list[DiverseVulSample]:
        """
        Filter samples by vulnerability target (0 for safe, 1 for vulnerable).

        Args:
            target: Target value to filter by

        Returns:
            List of filtered samples
        """
        return [sample for sample in self.samples if sample.target == target]

    def filter_by_cwe(self, cwe_list: list[str]) -> list[DiverseVulSample]:
        """
        Filter samples by CWE identifiers.

        Args:
            cwe_list: List of CWE identifiers to filter by

        Returns:
            List of samples that contain any of the specified CWEs
        """
        return [
            sample
            for sample in self.samples
            if any(cwe in sample.cwe for cwe in cwe_list)
        ]

    def filter_by_project(self, projects: list[str]) -> list[DiverseVulSample]:
        """
        Filter samples by project names.

        Args:
            projects: List of project names to filter by

        Returns:
            List of samples from the specified projects
        """
        return [sample for sample in self.samples if sample.project in projects]

    def get_statistics(self) -> dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary containing various statistics about the dataset
        """
        if not self.samples:
            return {}

        vulnerable_count = len([s for s in self.samples if s.target == 1])
        safe_count = len([s for s in self.samples if s.target == 0])

        # Get unique CWEs
        all_cwes = set()
        for sample in self.samples:
            all_cwes.update(sample.cwe)

        # Get unique projects
        projects = {sample.project for sample in self.samples}

        # Function size statistics
        sizes = [sample.size for sample in self.samples]

        return {
            "total_samples": len(self.samples),
            "vulnerable_samples": vulnerable_count,
            "safe_samples": safe_count,
            "vulnerability_ratio": (
                vulnerable_count / len(self.samples) if self.samples else 0
            ),
            "unique_cwes": len(all_cwes),
            "cwe_list": sorted(all_cwes),
            "unique_projects": len(projects),
            "project_list": sorted(projects),
            "min_function_size": min(sizes) if sizes else 0,
            "max_function_size": max(sizes) if sizes else 0,
            "avg_function_size": sum(sizes) / len(sizes) if sizes else 0,
            "metadata_available": len(self.metadata) > 0,
            "metadata_entries": len(self.metadata),
        }

    def save_processed_dataset(
        self,
        output_path: str | Path,
        samples: list[DiverseVulSample] | None = None,
    ) -> None:
        """
        Save processed dataset to JSON file.

        Args:
            output_path: Path to save the processed dataset
            samples: Optional list of samples to save (defaults to all loaded samples)
        """
        output_path = Path(output_path)
        samples_to_save = samples or self.samples

        if not samples_to_save:
            logger.warning("No samples to save")
            return

        # Convert samples to serializable format
        serializable_data = []
        for sample in samples_to_save:
            serializable_data.append(
                {
                    "func": sample.func,
                    "target": sample.target,
                    "cwe": sample.cwe,
                    "project": sample.project,
                    "commit_id": sample.commit_id,
                    "hash": sample.hash_value,
                    "size": sample.size,
                    "message": sample.message,
                },
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_json(serializable_data, output_path)

    def convert_to_pyg_graphs(
        self,
        dataset_samples: list[DiverseVulSample],
    ) -> list[Data]:
        """
        Convert dataset samples to PyTorch Geometric Data objects.

        Args:
            dataset_samples: List of DiverseVulSample objects to convert

        Returns:
            List of PyG Data objects
        """
        graphs = []
        failed_samples = 0

        # Initialize components
        ast_parser = ASTParser(language="c")
        feature_extractor = ASTFeatureExtractor()
        graph_builder = GraphBuilder()

        print(
            f"Converting {len(dataset_samples)} samples to PyG graphsgnn_vuln_detection..",
        )

        for i, sample in enumerate(dataset_samples):
            result = self._process_single_sample(
                sample,
                ast_parser,
                feature_extractor,
                graph_builder,
            )

            if result is not None:
                graphs.append(result)
            else:
                failed_samples += 1

            self._print_progress(i, len(dataset_samples), len(graphs))

        print(f"Conversion complete: {len(graphs)} successful, {failed_samples} failed")
        return graphs

    def _process_single_sample(
        self,
        sample: DiverseVulSample,
        ast_parser: ASTParser,
        feature_extractor: ASTFeatureExtractor,
        graph_builder: GraphBuilder,
    ) -> Data | None:
        """Process a single sample and convert it to PyG Data object."""
        try:
            # Extract code and label
            code = sample.func
            label = sample.target  # 0 for safe, 1 for vulnerable

            # Skip empty or invalid code
            if not code or len(code.strip()) == 0:
                return None

            # Parse code to AST
            ast_root = ast_parser.parse_code_to_ast(code)
            if ast_root is None:
                return None

            # Extract features
            node_features, node_mapping = feature_extractor.extract_features_from_ast(
                ast_root,
            )
            if len(node_features) == 0:
                return None

            # Build graph structure
            edge_index = graph_builder.build_ast_edges(ast_root, node_mapping)
            edge_index = self._ensure_valid_edges(edge_index, len(node_features))

            # Create PyG Data object
            data = self._create_pyg_data(node_features, edge_index, label, sample)

            # Validate and return
            return data if self._validate_pyg_data(data) else None

        except Exception as e:
            print(f"Error processing sample: {e}")
            return None

    def _ensure_valid_edges(self, edge_index, num_nodes: int):
        """Ensure edge_index is in the correct format."""
        # Handle case with no edges (create self-loops)
        if edge_index is None or edge_index.shape[1] == 0:
            edge_index = (
                torch.tensor([[i, i] for i in range(num_nodes)]).t().contiguous()
            )

        # Ensure proper edge format
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        return edge_index

    def _create_pyg_data(
        self,
        node_features: list,
        edge_index,
        label: int,
        sample: DiverseVulSample,
    ) -> Data:
        """Create PyTorch Geometric Data object from components."""
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.cwe = sample.cwe
        data.project = sample.project
        data.size = sample.size

        return data

    def _print_progress(
        self, current_idx: int, total_samples: int, successful_graphs: int,
    ) -> None:
        """Print progress indicator."""
        if (current_idx + 1) % 100 == 0 or current_idx == total_samples - 1:
            success_rate = successful_graphs / (current_idx + 1) * 100
            print(
                f"Processed {current_idx + 1}/{total_samples} samples, "
                f"created {successful_graphs} graphs (success rate: {success_rate:.1f}%)",
            )


def load_diversevul_dataset(
    dataset_path: str | Path,
    metadata_path: str | Path | None = None,
) -> DiverseVulDatasetLoader:
    """
    Load the DiverseVul dataset from the specified path.

    Args:
        dataset_path: Path to the DiverseVul dataset JSON file
        metadata_path: Optional path to the metadata JSON file

    Returns:
        DiverseVulDatasetLoader instance with loaded data
    """
    loader = DiverseVulDatasetLoader(dataset_path, metadata_path)
    loader.load_dataset()
    if metadata_path:
        loader.load_metadata()
    return loader


def load_megavul_dataset(
    dataset_path: str | Path,
    graph_dir_path: str | Path,
) -> MegaVulDatasetLoader:
    """
    Load the MegaVul dataset from the specified path.

    Args:
        dataset_path: Path to the MegaVul dataset JSON file (megavul_simple.json)
        graph_dir_path: Path to the directory containing graph JSON files

    Returns:
        MegaVulDatasetLoader instance with loaded data
    """
    loader = MegaVulDatasetLoader(dataset_path, graph_dir_path)
    loader.load_dataset()
    return loader
