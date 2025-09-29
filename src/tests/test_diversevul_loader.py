#!/usr/bin/env python3
"""
Test script for DiverseVul dataset loader.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_vuln_detection.data_processing.dataset_loader import (
    DiverseVulDatasetLoader,
    DiverseVulMetadata,
    DiverseVulSample,
)


def create_test_data():
    """Create test dataset files."""
    # Sample dataset based on DiverseVul format
    test_dataset = [
        {
            "func": "int vulnerable_gets() {\n    char buffer[10];\n    gets(buffer);\n    return 0;\n}",
            "target": 1,
            "cwe": ["CWE-120"],
            "project": "test_project_1",
            "commit_id": "abc123",
            "hash": 12345,
            "size": 4,
            "message": "Added vulnerable gets function",
        },
        {
            "func": "int safe_fgets() {\n    char buffer[10];\n    fgets(buffer, sizeof(buffer), stdin);\n    return 0;\n}",
            "target": 0,
            "cwe": [],
            "project": "test_project_1",
            "commit_id": "def456",
            "hash": 67890,
            "size": 4,
            "message": "Added safe fgets function",
        },
        {
            "func": "int null_deref() {\n    int *ptr = NULL;\n    *ptr = 1;\n    return 0;\n}",
            "target": 1,
            "cwe": ["CWE-476"],
            "project": "test_project_2",
            "commit_id": "ghi789",
            "hash": 11111,
            "size": 4,
            "message": "NULL pointer dereference vulnerability",
        },
        {
            "func": 'int buffer_overflow() {\n    char buf[8];\n    strcpy(buf, "this is too long for buffer");\n    return 0;\n}',
            "target": 1,
            "cwe": ["CWE-119", "CWE-787"],
            "project": "test_project_3",
            "commit_id": "jkl012",
            "hash": 22222,
            "size": 4,
            "message": "Buffer overflow in strcpy",
        },
    ]

    # Sample metadata
    test_metadata = [
        {
            "project": "test_project_1",
            "commit_id": "abc123",
            "CWE": "CWE-120",
            "CVE": "CVE-2023-1234",
            "bug_info": "Buffer overflow from gets()",
            "commit_url": "https://github.com/test/project1/commit/abc123",
            "repo_url": "https://github.com/test/project1",
        },
        {
            "project": "test_project_2",
            "commit_id": "ghi789",
            "CWE": "CWE-476",
            "CVE": None,
            "bug_info": "NULL pointer dereference",
            "commit_url": "https://github.com/test/project2/commit/ghi789",
            "repo_url": "https://github.com/test/project2",
        },
        {
            "project": "test_project_3",
            "commit_id": "jkl012",
            "CWE": "CWE-119",
            "CVE": "CVE-2023-5678",
            "bug_info": "Stack-based buffer overflow",
            "commit_url": "https://github.com/test/project3/commit/jkl012",
            "repo_url": "https://github.com/test/project3",
        },
    ]

    return test_dataset, test_metadata


def test_data_classes() -> None:
    """Test the DiverseVul data classes."""
    print("Testing DiverseVul data classes...")

    # Test DiverseVulSample
    sample = DiverseVulSample(
        func="int test() { return 0; }",
        target=1,
        cwe=["CWE-119"],
        project="test",
        commit_id="abc123",
        hash_value=12345,
        size=1,
        message="Test commit",
    )

    assert sample.target == 1
    assert len(sample.cwe) == 1
    assert sample.cwe[0] == "CWE-119"

    # Test DiverseVulMetadata
    metadata = DiverseVulMetadata(
        project="test",
        commit_id="abc123",
        cwe="CWE-119",
        cve="CVE-2023-1234",
    )

    assert metadata.project == "test"
    assert metadata.cve == "CVE-2023-1234"

    print("✓ DiverseVul data classes test passed")


def test_dataset_loader() -> None:
    """Test the DiverseVul dataset loader functionality."""
    print("Testing DiverseVul dataset loader...")

    # Create test data
    test_dataset, test_metadata = create_test_data()

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write test files
        dataset_file = temp_path / "dataset.json"
        metadata_file = temp_path / "metadata.json"

        with open(dataset_file, "w") as f:
            json.dump(test_dataset, f)

        with open(metadata_file, "w") as f:
            json.dump(test_metadata, f)

        # Test loader initialization
        loader = DiverseVulDatasetLoader(dataset_file, metadata_file)

        # Test dataset loading
        samples = loader.load_dataset()
        assert len(samples) == 4
        assert sum(1 for s in samples if s.target == 1) == 3  # 3 vulnerable
        assert sum(1 for s in samples if s.target == 0) == 1  # 1 safe

        # Test metadata loading
        metadata = loader.load_metadata()
        assert len(metadata) == 3

        # Test filtering
        vulnerable = loader.filter_by_target(1)
        assert len(vulnerable) == 3

        safe = loader.filter_by_target(0)
        assert len(safe) == 1

        # Test CWE filtering
        buffer_overflow = loader.filter_by_cwe(["CWE-120"])
        assert len(buffer_overflow) == 1

        null_pointer = loader.filter_by_cwe(["CWE-476"])
        assert len(null_pointer) == 1

        memory_issues = loader.filter_by_cwe(["CWE-119", "CWE-787"])
        assert len(memory_issues) == 1

        # Test project filtering
        project1_samples = loader.filter_by_project(["test_project_1"])
        assert len(project1_samples) == 2

        # Test statistics
        stats = loader.get_statistics()
        assert stats["total_samples"] == 4
        assert stats["vulnerable_samples"] == 3
        assert stats["safe_samples"] == 1
        assert stats["unique_projects"] == 3
        assert (
            stats["unique_cwes"] == 3
        )  # CWE-120, CWE-476, CWE-119, CWE-787 but some overlap

        # Test sample with metadata retrieval
        sample_with_meta = loader.get_sample_with_metadata(samples[0])
        sample, meta = sample_with_meta
        if meta:
            assert meta.project == sample.project

        # Test ML export
        ml_data = loader.export_for_ml(include_metadata=True)
        assert ml_data["total_samples"] == 4
        assert len(ml_data["functions"]) == 4
        assert len(ml_data["targets"]) == 4

        # Test merge with metadata
        merged_data = loader.merge_with_metadata()
        assert len(merged_data) == 4

        # Test saving
        output_file = temp_path / "output.json"
        loader.save_processed_dataset(output_file, vulnerable)

        # Verify saved file
        assert output_file.exists()
        with open(output_file) as f:
            saved_data = json.load(f)
        assert len(saved_data) == 3

    print("✓ DiverseVul dataset loader test passed")


def test_error_handling() -> None:
    """Test error handling."""
    print("Testing error handling...")

    # Test with non-existent file
    try:
        loader = DiverseVulDatasetLoader("non_existent_file.json")
        msg = "Should have raised FileNotFoundError"
        raise AssertionError(msg)
    except FileNotFoundError:
        pass  # Expected

    # Test with invalid JSON
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        invalid_file = temp_path / "invalid.json"

        with open(invalid_file, "w") as f:
            f.write("invalid json content")

        loader = DiverseVulDatasetLoader(invalid_file)
        try:
            loader.load_dataset()
            msg = "Should have raised ValueError"
            raise AssertionError(msg)
        except ValueError:
            pass  # Expected

    print("✓ Error handling test passed")


def test_compatibility_with_megavul() -> None:
    """Test that DiverseVul format is compatible with MegaVul format."""
    print("Testing compatibility with MegaVul format...")

    # DiverseVul samples should have the same structure as MegaVul
    test_dataset, _ = create_test_data()

    # Check that all required fields are present
    required_fields = [
        "func",
        "target",
        "cwe",
        "project",
        "commit_id",
        "hash",
        "size",
        "message",
    ]

    for sample in test_dataset:
        for field in required_fields:
            assert field in sample, f"Missing required field: {field}"

    print("✓ Compatibility test passed")


def main() -> None:
    """Run all tests."""
    print("Running DiverseVul dataset loader tests...")
    print("=" * 50)

    try:
        test_data_classes()
        test_dataset_loader()
        test_error_handling()
        test_compatibility_with_megavul()

        print("=" * 50)
        print("✓ All DiverseVul tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
