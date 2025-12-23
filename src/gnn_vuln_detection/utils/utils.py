"""Main utility module."""

import json
import logging
import re
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_pos_weight(dataset_iterable, num_classes: int):
    """
    dataset_iterable: iterable of torch_geometric.data.Data or batches
    data.y shape: [C], [1, C], or [B, C]

    returns torch.tensor of shape [num_classes]
    """

    pos = np.zeros(num_classes, dtype=np.int64)
    total = 0

    for data in dataset_iterable:
        y = data.y

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # Normalize shape to [N, C]
        if y.ndim == 1:
            y = y[None, :]  # [C] -> [1, C]
        elif y.ndim != 2:
            raise ValueError(f"Unexpected y shape: {y.shape}")

        # y is now [N, C]
        pos += (y == 1).sum(axis=0)
        total += y.shape[0]

    neg = total - pos

    pos_weight = np.ones(num_classes, dtype=np.float32)
    mask = pos > 0
    pos_weight[mask] = neg[mask] / pos[mask]

    return torch.tensor(pos_weight, dtype=torch.float)


def parse_git_url(git_url: str) -> tuple[str | None, str | None]:
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
    except OSError:
        logging.exception("Could not write to JSON file: %s", file_path)
        return 0
    return 1
