"""Dataset loading utilities for GNN-based vulnerability detection"""

import abc
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from gnn_vuln_detection.code_representation.code_representation import (
    CodeMetadata,
    CodeSample,
)
from gnn_vuln_detection.utils.file_loader import load_json
from gnn_vuln_detection.utils.utils import parse_git_url, save_json

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self.samples: list[CodeSample] = []
        if not self.dataset_path.exists():
            msg = f"Dataset file not found at {self.dataset_path}"
            raise FileNotFoundError(msg)

    @abc.abstractmethod
    def _create_pyg_data(
        self,
        node_features: list,
        edge_index,
        label: int,
        sample: CodeSample,
    ) -> Data:
        """Create PyTorch Geometric Data object from components."""
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        # data.cve = sample.cve_id
        data.cwe = sample.cwe_ids
        data.project = sample.metadata.project
        data.commit_id = sample.metadata.commit_id
        # data.size = sample.size

        return data

    @abc.abstractmethod
    def load_dataset(self) -> list[dict]:
        """Load the dataset from the JSON file."""

    def _serialize_data(self, samples: list[CodeSample]) -> list[dict[str, Any]]:
        """Serialize the dataset samples to a format suitable for saving."""

        return [asdict(sample) for sample in samples]

    def _print_progress(
        self,
        current_idx: int,
        total_samples: int,
        successful_graphs: int,
    ) -> None:
        """Print progress indicator."""
        if (current_idx + 1) % 100 == 0 or current_idx == total_samples - 1:
            success_rate = successful_graphs / (current_idx + 1) * 100
            logger.info(
                f"Processed {current_idx + 1}/{total_samples} samples, "
                f"created {successful_graphs} graphs (success rate: {success_rate:.1f}%)",
            )

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

    def save_processed_dataset(
        self,
        output_path: str | Path,
        samples_to_save: list[CodeSample] | None = None,
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

    @property
    def statistics(self) -> dict:
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
            }

        vulnerable_count = sum(1 for s in self.samples if s.label)
        non_vulnerable_count = len(self.samples) - vulnerable_count

        all_cve_ids = sorted({s.cve_id for s in self.samples if s.cve_id})
        all_projects = sorted(
            {s.metadata.project for s in self.samples if s.metadata.project},
        )

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
        }


class MegaVulDatasetLoader(DatasetLoader):
    """Loader for MegaVul vulnerability dataset (megavul_simple.json format)."""

    def load_dataset(self) -> list[CodeSample]:
        """
        Load the main dataset from JSON file.

        Returns:
            List of CodeSample objects
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
                    project, commit_id = parse_git_url(git_url)
                elif git_url is not None:
                    logger.warning(
                        f"git_url for item {i} is not a string: {git_url}. Cannot parse project/commit.",
                    )

                # func_before should only be populated if is_vul is True and func_before exists
                is_vul_val = item["is_vul"]
                # func_before_val = None
                # if is_vul_val:
                #     func_before_val = item.get("func_before")

                sample = CodeSample(
                    label=1 if is_vul_val else 0,
                    code=item["func"],  # 'func' key in JSON is the fixed function
                    cve_id=item.get("cve_id"),
                    cwe_ids=item.get("cwe_ids"),
                    metadata=CodeMetadata(
                        project=project or "",
                        repo_url=(git_url if isinstance(git_url, str) else None),
                        commit_id=commit_id or "",
                        bug_info=item.get("commit_msg"),
                        file_path=item.get("file_path"),
                    ),
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
            "Successfully loaded %d samples. Skipped %d samples.",
            len(self.samples),
            skipped_samples_count,
        )
        return self.samples


class DiverseVulDatasetLoader(DatasetLoader):
    """Loader for DiverseVul vulnerability dataset (diversevul.json format)."""

    def load_dataset(self) -> list[CodeSample]:
        """
        Load the main dataset from JSON file.

        Returns:
            List of CodeSample objects
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
                cwes = item.get("cwe", [])
                sample = CodeSample(
                    label=1 if cwes else 0,
                    code=item["func"],
                    cwe_ids=cwes,
                    metadata=CodeMetadata(
                        project=item.get("project", ""),
                        commit_id=item.get("commit_id", ""),
                        bug_info=item.get("message"),
                    ),
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
            "Successfully loaded %d samples. Skipped %d samples.",
            len(self.samples),
            skipped_samples_count,
        )
        return self.samples


    def load_metadata(self) -> dict[str, CodeMetadata]:
        """
        Load metadata from JSON file if available.

        Returns:
            Dictionary mapping commit_id to CodeMetadata
        """
        # try:
        #     with self.metadata_path.open("r", encoding="utf-8") as f:
        #         raw_metadata = json.load(f)
        # except json.JSONDecodeError:
        #     logger.exception("Invalid JSON in metadata file:")
        #     return {}
        # except UnicodeDecodeError:
        #     logger.exception("Encoding error in metadata file:")
        #     return {}

        # if not isinstance(raw_metadata, list):
        #     logger.error("Metadata JSON should contain a list of entries")
        #     return {}

        self.metadata = {}
        # for item in raw_metadata:
        #     try:
        #         metadata = CodeMetadata(
        #             project=item["project"],
        #             commit_id=item["commit_id"],
        #             bug_info=item.get("bug_info"),
        #             commit_url=item.get("commit_url"),
        #             repo_url=item.get("repo_url"),
        #         )
        #     except KeyError as e:
        #         logger.warning(f"Skipping metadata entry due to missing key: {e}")
        #         continue

        logger.info(f"Loaded metadata for {len(self.metadata)} DiverseVul entries")
        return self.metadata
