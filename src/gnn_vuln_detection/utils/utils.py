"""Main utility module."""

import json
import logging
import re
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_pos_weight(dataset_list, num_classes: int):
    """
    dataset_list: iterable of torch_geometric.data.Data with data.y shape [1, num_classes]
    returns torch.tensor of shape [num_classes] for BCEWithLogitsLoss(pos_weight=...)
    pos_weight[c] = (neg_count / pos_count)  (if pos_count == 0 -> 1.0)
    """
    pos = np.zeros(num_classes, dtype=np.int64)
    total = 0
    for data in dataset_list:
        # ensure data.y is [1, C]
        y = data.y
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if y.ndim == 2:
            y = y.squeeze(0)
        pos += (y == 1).astype(np.int64)
        total += 1

    neg = total - pos
    pos_weight = []
    for p, n in zip(pos, neg, strict=False):
        if p == 0:
            pos_weight.append(1.0)
        else:
            # To avoid huge weights clamp to something reasonable if needed
            pos_weight.append(float(n) / float(p))
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
