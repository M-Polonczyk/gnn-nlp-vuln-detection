from dataclasses import dataclass, field
from enum import Enum


class LanguageEnum(Enum):
    C = "c"
    CPP = "cpp"
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    GO = "go"
    PHP = "php"


@dataclass
class CodeMetadata:
    """Data class representing metadata for a DiverseVul sample."""

    project: str = ""
    commit_id: str = ""
    bug_info: str | None = None
    commit_url: str | None = None
    repo_url: str | None = None
    file_path: str | None = None


@dataclass
class CodeSample:
    """Dataclass representing a code sample for vulnerability detection."""

    code: str
    label: int  # 0: safe, 1: vulnerable
    language: LanguageEnum = LanguageEnum.C
    cve_id: str | None = None
    # TODO: change to dict
    cwe_ids: list[str] | None = None
    cwe_ids_labeled: list[int] | None = None
    function_name: str | None = None
    line_numbers: tuple[int, int] | None = None
    size: int | None = None
    metadata: CodeMetadata = field(default_factory=CodeMetadata)
    graph: object | None = None  # Placeholder for graph representation
    features: dict | None = None  # Placeholder for extracted features
