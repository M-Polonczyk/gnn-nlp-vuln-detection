# Scripts Directory

This directory contains utility scripts for the GNN Vulnerability Detection project.

## Testing Scripts

### `run_tests.py`
Convenience script to run the main test suite. This is a wrapper around the main test runner in `src/tests/run_tests.py`.

**Usage:**
```bash
# From project root
python scripts/run_tests.py [OPTIONS]

# Examples
python scripts/run_tests.py --module ast_parser
python scripts/run_tests.py --coverage
python scripts/run_tests.py --list
```

### `ci_test.sh`
Comprehensive CI/CD test script that runs the full test suite with additional checks.

**Usage:**
```bash
# From project root
./scripts/ci_test.sh
```

This script includes:
- Python version check
- Dependency installation
- Syntax validation
- Import testing
- Full test suite execution
- Component-specific testing
- Coverage analysis (if available)

## Data Processing Scripts

### `preprocess_data.py`
Script for preprocessing vulnerability datasets.

### `train_model.py`
Script for training GNN models on vulnerability data.

### `evaluate_model.py`
Script for evaluating trained models on test datasets.

## Usage Tips

1. **Run from project root**: All scripts are designed to be run from the project root directory.

2. **Make commands**: Use the Makefile for convenience:
   ```bash
   make test-verify    # Run verification tests
   make test-ci        # Run CI test suite
   ```

3. **Script permissions**: Make scripts executable if needed:
   ```bash
   chmod +x scripts/*.py scripts/*.sh
   ```

4. **Virtual environment**: Ensure you're in the correct Python environment before running scripts.
