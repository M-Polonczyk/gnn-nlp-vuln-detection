#!/usr/bin/env python3
"""
Example usage of the MegaVul dataset loader.

This script demonstrates how to load and work with the MegaVul dataset
for vulnerability detection research.
"""

import json
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from gnn_vuln_detection.data_processing.dataset_loader import MegaVulDatasetLoader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the src directory structure is correct")
    sys.exit(1)


def display_statistics(stats) -> None:
    """Display dataset statistics."""
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Vulnerable samples: {stats['vulnerable_samples']:,}")
    print(f"Safe samples: {stats['safe_samples']:,}")
    print(f"Vulnerability ratio: {stats['vulnerability_ratio']:.2%}")
    print(f"Unique CWEs: {stats['unique_cwes']}")
    print(f"Unique projects: {stats['unique_projects']}")
    print(
        f"Function size range: {stats['min_function_size']} - {stats['max_function_size']} lines",
    )
    print(f"Average function size: {stats['avg_function_size']:.1f} lines")

    # Show some example CWEs and projects
    if stats["cwe_list"]:
        print(f"\nTop 10 CWEs: {', '.join(stats['cwe_list'][:10])}")
    if stats["project_list"]:
        print(f"Top 10 projects: {', '.join(stats['project_list'][:10])}")


def demonstrate_filtering(loader, stats):
    """Demonstrate dataset filtering capabilities."""
    print("\n" + "=" * 50)
    print("FILTERING EXAMPLES")
    print("=" * 50)

    vulnerable_samples = loader.filter_by_target(1)
    safe_samples = loader.filter_by_target(0)
    print(f"Vulnerable samples: {len(vulnerable_samples)}")
    print(f"Safe samples: {len(safe_samples)}")

    # Example: Filter by specific CWEs
    if stats["cwe_list"]:
        example_cwe = stats["cwe_list"][0] if stats["cwe_list"] else None
        if example_cwe:
            cwe_samples = loader.filter_by_cwe([example_cwe])
            print(f"Samples with {example_cwe}: {len(cwe_samples)}")

    # Example: Filter by project
    if stats["project_list"]:
        example_project = stats["project_list"][0]
        project_samples = loader.filter_by_project([example_project])
        print(f"Samples from {example_project}: {len(project_samples)}")

    return vulnerable_samples


def show_sample_function(loader, samples) -> None:
    """Display a sample function with metadata."""
    if not samples:
        return

    print("\n" + "=" * 50)
    print("SAMPLE FUNCTION")
    print("=" * 50)
    sample = samples[0]
    print(f"Project: {sample.project}")
    print(f"Vulnerable: {'Yes' if sample.target == 1 else 'No'}")
    print(f"CWEs: {', '.join(sample.cwe) if sample.cwe else 'None'}")
    print(f"Size: {sample.size} lines")
    print(f"Commit: {sample.commit_id}")
    print(f"Message: {sample.message}")
    print("\nFunction code (first 500 chars):")
    print("-" * 30)
    print(sample.func[:500])
    if len(sample.func) > 500:
        print("... (truncated)")

    # Show metadata if available
    sample_with_metadata = loader.get_sample_with_metadata(sample)
    if sample_with_metadata[1]:
        metadata = sample_with_metadata[1]
        print("\nMetadata:")
        print(f"  CVE: {metadata.cve or 'N/A'}")
        print(f"  Bug info: {metadata.bug_info or 'N/A'}")
        print(f"  Commit URL: {metadata.commit_url or 'N/A'}")


def save_sample_data(loader, vulnerable_samples) -> None:
    """Save a subset of the data."""
    print("\n" + "=" * 50)
    print("SAVING EXAMPLE")
    print("=" * 50)

    # Save only vulnerable samples to a new file
    output_path = Path("data/processed/vulnerable_samples.json")
    if vulnerable_samples:
        loader.save_processed_dataset(
            output_path, vulnerable_samples[:100],
        )  # Save first 100
        print(
            f"Saved {min(100, len(vulnerable_samples))} vulnerable samples to {output_path}",
        )


def main() -> None:
    """Main function demonstrating MegaVul dataset usage."""

    # Example paths - adjust these to match your actual data locations
    dataset_path = "data/megavul/dataset.json"  # Main dataset file
    metadata_path = "data/megavul/metadata.json"  # Optional metadata file

    # Check if files exist
    if not Path(dataset_path).exists():
        print(f"Dataset file not found at {dataset_path}")
        print("Please download the dataset from https://github.com/Icyrockton/MegaVul")
        print("Expected format: JSON file with list of function samples")
        print("Run create_sample_data() to generate test data")
        return

    # Initialize the loader
    print("Initializing MegaVul dataset loader...")
    loader = MegaVulDatasetLoader(
        dataset_path=dataset_path,
        metadata_path=metadata_path if Path(metadata_path).exists() else None,
    )

    # Load the main dataset
    print("Loading dataset...")
    samples = loader.load_dataset()
    print(f"Loaded {len(samples)} samples")

    # Load metadata if available
    if loader.metadata_path:
        print("Loading metadata...")
        metadata = loader.load_metadata()
        print(f"Loaded metadata for {len(metadata)} entries")

    # Display dataset statistics
    stats = loader.get_statistics()
    display_statistics(stats)

    # Demonstrate filtering
    vulnerable_samples = demonstrate_filtering(loader, stats)

    # Show sample function
    show_sample_function(loader, samples)

    # Save sample data
    save_sample_data(loader, vulnerable_samples)

    print("\nDataset loading complete!")


def create_sample_data() -> None:
    """Create sample data files for testing (based on the provided examples)."""

    # Sample dataset entries based on your examples
    sample_dataset = [
        {
            "func": "int _gnutls_ciphertext2compressed(gnutls_session_t session,\n\t\t\t\t opaque * compress_data,\n\t\t\t\t int compress_size,\n\t\t\t\t gnutls_datum_t ciphertext, uint8 type)\n{\n uint8 MAC[MAX_HASH_SIZE];\n uint16 c_length;\n uint8 pad;\n int length;\n // ... rest of function ...\n return length;\n}",
            "target": 1,
            "cwe": [],
            "project": "gnutls",
            "commit_id": "7ad6162573ba79a4392c63b453ad0220ca6c5ace",
            "hash": 73008646937836648589283922871188272089,
            "size": 157,
            "message": "added an extra check while checking the padding.",
        },
        {
            "func": 'static char *make_filename_safe(const char *filename TSRMLS_DC)\n{\n\tif (*filename && strncmp(filename, ":memory:", sizeof(":memory:")-1)) {\n\t\tchar *fullpath = expand_filepath(filename, NULL TSRMLS_CC);\n\t\t// ... rest of function ...\n\t}\n\treturn estrdup(filename);\n}',
            "target": 1,
            "cwe": ["CWE-264"],
            "project": "php-src",
            "commit_id": "055ecbc62878e86287d742c7246c21606cee8183",
            "hash": 211824207069112513181516095447837228041,
            "size": 22,
            "message": "Improve check for :memory: pseudo-filename in SQlite",
        },
    ]

    # Sample metadata entries
    sample_metadata = [
        {
            "project": "libass",
            "commit_id": "017137471d0043e0321e377ed8da48e45a3ec632",
            "CWE": "CWE-369",
            "CVE": None,
            "bug_info": "Out-of-bounds Read",
            "commit_url": "https://github.com/libass/libass/commit/017137471d0043e0321e377ed8da48e45a3ec632",
            "repo_url": "https://github.com/libass/libass",
        },
        {
            "project": "gpac",
            "commit_id": "00194f5fe462123f70b0bae7987317b52898b868",
            "CWE": "CWE-476",
            "CVE": None,
            "bug_info": "NULL Pointer Dereference",
            "commit_url": "https://github.com/gpac/gpac/commit/00194f5fe462123f70b0bae7987317b52898b868",
            "repo_url": "https://github.com/gpac/gpac",
        },
    ]

    # Create directories
    data_dir = Path("data/megavul")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save sample files
    with open(data_dir / "dataset.json", "w") as f:
        json.dump(sample_dataset, f, indent=2)

    with open(data_dir / "metadata.json", "w") as f:
        json.dump(sample_metadata, f, indent=2)

    print(f"Created sample data files in {data_dir}/")
    print("- dataset.json: Main dataset with function samples")
    print("- metadata.json: Metadata for vulnerability analysis")


if __name__ == "__main__":
    # Uncomment the line below to create sample data for testing
    # create_sample_data()

    main()
