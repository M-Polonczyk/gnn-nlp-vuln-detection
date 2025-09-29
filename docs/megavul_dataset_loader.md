# MegaVul Dataset Loader

This module provides a comprehensive dataset loader for the [MegaVul](https://github.com/Icyrockton/MegaVul) vulnerability dataset, which contains vulnerable and non-vulnerable C/C++ functions from various open-source projects.

## Features

- **Structured Data Loading**: Load function samples with metadata including CWE classifications, project information, and vulnerability status
- **Metadata Integration**: Optional metadata loading with CVE information, bug details, and repository links
- **Filtering Capabilities**: Filter samples by vulnerability status, CWE types, or projects
- **Statistics Generation**: Comprehensive dataset statistics and analysis
- **Data Validation**: Built-in validation for data integrity
- **Export Functionality**: Save processed datasets in JSON format

## Dataset Format

### Main Dataset Structure

Each sample in the dataset contains:

```json
{
  "func": "function source code as string",
  "target": 1,  // 0 for non-vulnerable, 1 for vulnerable
  "cwe": ["CWE-XXX", "CWE-YYY"],  // List of CWE identifiers
  "project": "project_name",
  "commit_id": "git_commit_hash",
  "hash": 123456789,  // Function hash
  "size": 50,  // Function size in lines
  "message": "commit message"
}
```

### Metadata Structure

Optional metadata provides additional context:

```json
{
  "project": "project_name",
  "commit_id": "git_commit_hash",
  "CWE": "CWE-XXX",
  "CVE": "CVE-YYYY-XXXXX",
  "bug_info": "Description of the vulnerability",
  "commit_url": "https://github.com/owner/repo/commit/hash",
  "repo_url": "https://github.com/owner/repo"
}
```

## Usage

### Basic Usage

```python
from gnn_vuln_detection.data_processing.dataset_loader import MegaVulDatasetLoader

# Initialize loader
loader = MegaVulDatasetLoader(
    dataset_path="data/megavul/dataset.json",
    metadata_path="data/megavul/metadata.json"  # Optional
)

# Load dataset
samples = loader.load_dataset()
metadata = loader.load_metadata()  # If metadata file provided

print(f"Loaded {len(samples)} samples")
```

### Dataset Statistics

```python
# Get comprehensive statistics
stats = loader.get_statistics()

print(f"Total samples: {stats['total_samples']}")
print(f"Vulnerable samples: {stats['vulnerable_samples']}")
print(f"Vulnerability ratio: {stats['vulnerability_ratio']:.2%}")
print(f"Unique CWEs: {stats['unique_cwes']}")
print(f"Unique projects: {stats['unique_projects']}")
```

### Filtering Data

```python
# Filter by vulnerability status
vulnerable_samples = loader.filter_by_target(1)  # Vulnerable functions
safe_samples = loader.filter_by_target(0)       # Non-vulnerable functions

# Filter by specific CWEs
buffer_overflow_samples = loader.filter_by_cwe(["CWE-119", "CWE-120"])

# Filter by projects
specific_project_samples = loader.filter_by_project(["linux", "openssl"])
```

### Working with Metadata

```python
# Get sample with its metadata
for sample in samples[:10]:
    sample_data, metadata = loader.get_sample_with_metadata(sample)
    
    print(f"Function from {sample_data.project}")
    print(f"Vulnerable: {sample_data.target == 1}")
    
    if metadata:
        print(f"CVE: {metadata.cve}")
        print(f"Bug type: {metadata.bug_info}")
        print(f"Commit URL: {metadata.commit_url}")
```

### Saving Processed Data

```python
# Save filtered dataset
vulnerable_functions = loader.filter_by_target(1)
loader.save_processed_dataset(
    "data/processed/vulnerable_only.json", 
    vulnerable_functions
)

# Save specific CWE types
memory_issues = loader.filter_by_cwe(["CWE-119", "CWE-120", "CWE-787"])
loader.save_processed_dataset(
    "data/processed/memory_vulnerabilities.json",
    memory_issues
)
```

## Data Classes

### MegaVulSample

Represents a single function sample with validation:

```python
@dataclass
class MegaVulSample:
    func: str           # Function source code
    target: int         # Vulnerability status (0/1)
    cwe: List[str]      # CWE identifiers
    project: str        # Project name
    commit_id: str      # Git commit hash
    hash_value: int     # Function hash
    size: int           # Function size in lines
    message: str        # Commit message
```

### MegaVulMetadata

Represents metadata for additional context:

```python
@dataclass
class MegaVulMetadata:
    project: str                    # Project name
    commit_id: str                  # Git commit hash
    cwe: Optional[str] = None       # Primary CWE
    cve: Optional[str] = None       # CVE identifier
    bug_info: Optional[str] = None  # Bug description
    commit_url: Optional[str] = None # Commit URL
    repo_url: Optional[str] = None   # Repository URL
```

## Example Application

See `examples/megavul_dataset_usage.py` for a complete example that demonstrates:

- Loading the dataset and metadata
- Displaying comprehensive statistics
- Filtering data by various criteria
- Showing sample functions with their metadata
- Saving processed subsets

## Dataset Download

To use this loader, first download the MegaVul dataset:

1. Visit the [MegaVul repository](https://github.com/Icyrockton/MegaVul)
2. Download the dataset and metadata JSON files
3. Place them in your data directory
4. Update the paths in your configuration

## Configuration

Update `config/dataset_paths.yml` with your dataset locations:

```yaml
megavul:
  dataset_path: data/raw/megavul/dataset.json
  metadata_path: data/raw/megavul/metadata.json
  processed_path: data/processed/megavul/
```

## Error Handling

The loader includes comprehensive error handling:

- **File validation**: Checks for file existence and readability
- **JSON validation**: Validates JSON structure and content
- **Data validation**: Ensures required fields are present and valid
- **Encoding handling**: Handles various text encodings
- **Graceful degradation**: Continues processing when encountering invalid samples

## Logging

The loader uses Python's logging module to provide detailed information:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now you'll see detailed loading progress and statistics
loader = MegaVulDatasetLoader(dataset_path, metadata_path)
samples = loader.load_dataset()
```

## Integration with GNN Models

This loader is designed to integrate seamlessly with the vulnerability detection pipeline:

1. **Data Loading**: Use the loader to get structured function samples
2. **AST Parsing**: Pass function code to AST parsers for graph generation
3. **Feature Extraction**: Extract features from both code and metadata
4. **Model Training**: Use filtered datasets for training GNN models
5. **Evaluation**: Use stratified sampling based on CWE types or projects

## Contributing

When adding new features to the dataset loader:

1. Maintain backward compatibility
2. Add comprehensive docstrings
3. Include error handling
4. Add logging for debugging
5. Write tests for new functionality
6. Update this documentation
