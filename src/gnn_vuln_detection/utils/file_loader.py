from pathlib import Path
from typing import Any

import yaml


def load_file(file_path: str | Path) -> str:
    """
    Load a file and return its content.

    Args:
        file_path (str or Path): Path to the file to be loaded.

    Returns:
        str: Content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
    """
    if not isinstance(file_path, (str, Path)):
        msg = "file_path must be a string or Path object"
        raise TypeError(msg)

    file_path = Path(file_path)

    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        msg = f"Error reading file {file_path}: {e}"
        raise OSError(msg) from e


def save_file(file_path: str | Path, content: str) -> None:
    """
    Save content to a file.

    Args:
        file_path (str or Path): Path to the file where content will be saved.
        content (str): Content to be saved in the file.

    Raises:
        IOError: If there is an error writing to the file.
    """
    if not isinstance(file_path, (str, Path)):
        msg = "file_path must be a string or Path object"
        raise TypeError(msg)

    file_path = Path(file_path)

    try:
        with file_path.open("w", encoding="utf-8") as f:
            f.write(content)
    except OSError:
        msg = f"Error writing to file {file_path}"
        raise OSError(msg) from e


def load_json(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Load a JSON file and return its content as a dictionary.

    Args:
        file_path (str or Path): Path to the JSON file to be loaded.

    Returns:
        dict: Content of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is not valid JSON.
    """
    import json

    content = load_file(file_path)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in file {file_path}: {e}"
        raise ValueError(msg) from e


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file and return its content as a dictionary.

    Args:
        file_path (str or Path): Path to the YAML file to be loaded.

    Returns:
        dict: Content of the YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is not valid YAML.
    """

    content = load_file(file_path)

    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        msg = f"Invalid YAML in file {file_path}: {e}"
        raise ValueError(msg) from e


def load_code(file_path: str | Path) -> str:
    """
    Load a code file and return its content.

    Args:
        file_path (str or Path): Path to the code file to be loaded.

    Returns:
        str: Content of the code file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
    """
    return load_file(file_path)
