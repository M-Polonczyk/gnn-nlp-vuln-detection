# DiverseVul Dataset Loader

This module provides a comprehensive dataset loader for the DiverseVul vulnerability dataset, which contains vulnerable and non-vulnerable C/C++ functions from various open-source projects. The DiverseVul dataset follows a similar format to MegaVul and is designed for vulnerability detection research.

## Features

- **Structured Data Loading**: Load function samples with metadata including CWE classifications, project information, and vulnerability status
- **Metadata Integration**: Optional metadata loading with CVE information, bug details, and repository links  
- **Filtering Capabilities**: Filter samples by vulnerability status, CWE types, or projects
- **Statistics Generation**: Comprehensive dataset statistics and analysis
- **ML Export**: Export data in formats suitable for machine learning training
- **Data Validation**: Built-in validation for data integrity
- **Export Functionality**: Save processed datasets in JSON format
- **MegaVul Compatibility**: Uses the same data format as MegaVul for interoperability

## Dataset Format

### Main Dataset Structure

Each sample in the DiverseVul dataset contains:

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
from gnn_vuln_detection.data_processing.dataset_loader import DiverseVulDatasetLoader

# Initialize loader
loader = DiverseVulDatasetLoader(
    dataset_path="data/diversevul/dataset.json",
    metadata_path="data/diversevul/metadata.json"  # Optional
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

### Machine Learning Export

The DiverseVul loader includes specialized functionality for ML training:

```python
# Export data for ML training
ml_data = loader.export_for_ml(include_metadata=True)

print(f"Functions: {len(ml_data['functions'])}")
print(f"Labels: {len(ml_data['targets'])}")
print(f"CWE lists: {len(ml_data['cwes'])}")

# Use the exported data for training
X = ml_data['functions']  # Function source code
y = ml_data['targets']    # Vulnerability labels
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

# Merge all samples with metadata
merged_data = loader.merge_with_metadata()
for sample, metadata in merged_data:
    # Process sample with its metadata
    pass
```

### Saving Processed Data

```python
# Save filtered dataset
vulnerable_functions = loader.filter_by_target(1)
loader.save_processed_dataset(
    "data/processed/diversevul_vulnerable_only.json", 
    vulnerable_functions
)

# Save specific CWE types
memory_issues = loader.filter_by_cwe(["CWE-119", "CWE-120", "CWE-787"])
loader.save_processed_dataset(
    "data/processed/diversevul_memory_vulnerabilities.json",
    memory_issues
)
```

## Data Classes

### DiverseVulSample

Represents a single function sample with validation:

```python
@dataclass
class DiverseVulSample:
    func: str           # Function source code
    target: int         # Vulnerability status (0/1)
    cwe: List[str]      # CWE identifiers
    project: str        # Project name
    commit_id: str      # Git commit hash
    hash_value: int     # Function hash
    size: int           # Function size in lines
    message: str        # Commit message
```

### DiverseVulMetadata

Represents metadata for additional context:

```python
@dataclass
class DiverseVulMetadata:
    project: str                    # Project name
    commit_id: str                  # Git commit hash
    cwe: Optional[str] = None       # Primary CWE
    cve: Optional[str] = None       # CVE identifier
    bug_info: Optional[str] = None  # Bug description
    commit_url: Optional[str] = None # Commit URL
    repo_url: Optional[str] = None   # Repository URL
```

## Example Applications

See `examples/diversevul_dataset_usage.py` for a complete example that demonstrates:

- Loading the dataset and metadata
- Displaying comprehensive statistics
- Filtering data by various criteria
- Exporting data for ML training
- Showing sample functions with their metadata
- Saving processed subsets

## Utility Functions

### Direct Dataset Loading

```python
from gnn_vuln_detection.data_processing.dataset_loader import load_diversevul_dataset

# Load dataset with one function call
loader = load_diversevul_dataset(
    dataset_path="data/diversevul/dataset.json",
    metadata_path="data/diversevul/metadata.json"
)

# Dataset and metadata are automatically loaded
samples = loader.samples
print(f"Loaded {len(samples)} samples")
```

## Configuration

Update `config/dataset_paths.yml` with your dataset locations:

```yaml
diversevul:
  dataset_path: data/raw/diversevul/dataset.json
  metadata_path: data/raw/diversevul/metadata.json
  processed_path: data/processed/diversevul/
```

## Comparison with MegaVul

The DiverseVul loader is designed to be compatible with the MegaVul loader:

| Feature | MegaVul | DiverseVul |
|---------|---------|------------|
| Data Format | JSON list of samples | JSON list of samples |
| Sample Fields | func, target, cwe, project, commit_id, hash, size, message | Same |
| Metadata Format | project, commit_id, CWE, CVE, bug_info, commit_url, repo_url | Same |
| Filtering | By target, CWE, project | Same |
| Statistics | Comprehensive stats | Same |
| ML Export | Basic export | Enhanced ML export |
| Merge Functionality | Basic | Enhanced with merge_with_metadata() |

## Advanced Features

### Enhanced ML Export

The DiverseVul loader provides additional ML-focused functionality:

```python
# Export with metadata for enriched features
ml_data = loader.export_for_ml(include_metadata=True)

# Access structured data
functions = ml_data['functions']
labels = ml_data['targets']
cwe_lists = ml_data['cwes']
projects = ml_data['projects']
sizes = ml_data['sizes']
metadata = ml_data['metadata']  # If include_metadata=True
```

### Batch Processing

```python
# Process samples in batches
batch_size = 1000
samples = loader.samples

for i in range(0, len(samples), batch_size):
    batch = samples[i:i+batch_size]
    # Process batch
    process_batch(batch)
```

## Error Handling

The loader includes comprehensive error handling:

- **File validation**: Checks for file existence and readability
- **JSON validation**: Validates JSON structure and content
- **Data validation**: Ensures required fields are present and valid
- **Encoding handling**: Handles various text encodings
- **Graceful degradation**: Continues processing when encountering invalid samples

## Testing

Run the test suite to verify functionality:

```bash
python tests/test_diversevul_loader.py
```

The test suite covers:
- Data class validation
- Dataset loading and parsing
- Filtering functionality
- Statistics generation
- ML export features
- Error handling
- Compatibility with MegaVul format

## Integration with GNN Models

This loader integrates seamlessly with the vulnerability detection pipeline:

1. **Data Loading**: Use the loader to get structured function samples
2. **AST Parsing**: Pass function code to AST parsers for graph generation
3. **Feature Extraction**: Extract features from both code and metadata
4. **Model Training**: Use filtered datasets for training GNN models
5. **Evaluation**: Use stratified sampling based on CWE types or projects

## Contributing

When adding new features to the DiverseVul dataset loader:

1. Maintain compatibility with MegaVul format
2. Add comprehensive docstrings
3. Include error handling and logging
4. Add tests for new functionality
5. Update this documentation
6. Consider ML workflow implications
