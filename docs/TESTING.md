# Testing Guide for GNN Vulnerability Detection

This document provides comprehensive information about testing in the GNN Vulnerability Detection project.

## Test Structure

The project uses a comprehensive testing framework with multiple ways to run tests:

```
src/tests/
├── __init__.py
├── run_tests.py              # Main test runner
├── test_ast_parser.py        # AST parser tests
├── test_naming_patterns.py   # Naming pattern tests  
├── test_naming_analyzer.py   # Naming analyzer tests
├── test_graph_builder.py     # Graph construction tests
├── test_models.py           # Model tests
├── test_diversevul_loader.py # DiverseVul dataset tests
└── test_megavul_loader.py    # MegaVul dataset tests
```

## Running Tests

### Quick Methods

1. **Use the convenience script** (from project root):
   ```bash
   python scripts/run_tests.py
   ```

2. **Use Make commands** (from project root):
   ```bash
   make test          # Run all tests
   make test-ast      # Run AST parser tests only
   make test-naming   # Run naming convention tests only
   make test-coverage # Run with coverage analysis
   make test-list     # List all available tests
   make test-verify   # Run verification tests
   make test-ci       # Run full CI test suite
   ```

3. **Use the CI script** (from project root):
   ```bash
   ./scripts/ci_test.sh
   ```

### Advanced Test Runner

The main test runner (`src/tests/run_tests.py`) provides extensive options:

```bash
# Basic usage
python src/tests/run_tests.py

# Run specific module tests
python src/tests/run_tests.py --module ast_parser
python src/tests/run_tests.py --module naming_patterns

# Run specific test class
python src/tests/run_tests.py --module ast_parser --class TestASTParser

# Different verbosity levels
python src/tests/run_tests.py --quiet       # Minimal output
python src/tests/run_tests.py --verbose     # More detailed output
python src/tests/run_tests.py -vv           # Very verbose

# Coverage analysis
python src/tests/run_tests.py --coverage

# List available tests
python src/tests/run_tests.py --list
```

### Using Pytest (Alternative)

If you prefer pytest, install it and run:

```bash
pip install pytest pytest-cov
pytest src/tests/
pytest src/tests/test_ast_parser.py -v
pytest --cov=src/gnn_vuln_detection --cov-report=html
```

## Test Categories

### 1. AST Parser Tests (`test_ast_parser.py`)

Tests the Abstract Syntax Tree parsing functionality:

- **TestASTParser**: Tests basic parsing functionality
- **TestASTParserCCode**: Tests C code parsing
- **TestASTParserPythonCode**: Tests Python code parsing  
- **TestASTParserErrorHandling**: Tests error handling and edge cases

Key test areas:
- Parsing different code constructs (functions, classes, variables)
- Handling syntax errors gracefully
- Extracting identifiers correctly
- Processing different programming languages

### 2. Naming Convention Tests (`test_naming_patterns.py`)

Tests the naming pattern recognition system:

- **TestNamingPatterns**: Tests pattern matching for different naming conventions
- **TestIdentifyNamingConvention**: Tests the main convention identification function
- **TestBadNames**: Tests detection of non-descriptive names

Tested patterns:
- `snake_case`
- `camelCase` 
- `PascalCase`
- `SCREAMING_SNAKE_CASE`
- `kebab-case`
- `lower.dot.case`

### 3. Naming Analyzer Tests (`test_naming_analyzer.py`)

Tests the comprehensive naming analysis system:

- **TestNamingAnalyzer**: Tests the main analyzer functionality
- **TestMeaningfulnessAnalysis**: Tests identifier meaningfulness detection
- **TestConsistencyChecking**: Tests naming consistency across code

## Writing New Tests

### Test File Structure

Follow this template for new test files:

```python
#!/usr/bin/env python3
import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestYourComponent(unittest.TestCase):
    """Test cases for YourComponent."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Your test code here
        self.assertTrue(True)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Your test code here
        pass

if __name__ == '__main__':
    unittest.main()
```

### Test Naming Conventions

- Test files: `test_component_name.py`
- Test classes: `TestComponentName`
- Test methods: `test_specific_functionality`
- Use descriptive test method names that explain what is being tested

### Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Clarity**: Test names should clearly indicate what is being tested
3. **Coverage**: Test both happy paths and edge cases
4. **Documentation**: Add docstrings to test classes and complex test methods
5. **Assertions**: Use appropriate assertion methods (`assertEqual`, `assertRaises`, etc.)

## Coverage Analysis

The project supports code coverage analysis to ensure comprehensive testing:

```bash
# Run with coverage (HTML report generated in coverage_html/)
python src/tests/run_tests.py --coverage

# Or with pytest
pytest --cov=src/gnn_vuln_detection --cov-report=html --cov-report=term
```

Coverage reports help identify:
- Untested code paths
- Missing edge case tests
- Areas needing more comprehensive testing

## Continuous Integration

The `scripts/ci_test.sh` script runs a complete test suite suitable for CI/CD:

1. Python version check
2. Dependency installation
3. Syntax validation
4. Import testing
5. Full test suite execution
6. Component-specific testing
7. Coverage analysis (if available)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running tests from the correct directory and that the Python path is set correctly.

2. **Missing Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test Discovery Issues**: Make sure test files follow the naming convention (`test_*.py`).

4. **Path Issues**: Use absolute paths when possible, especially in test configurations.

### Getting Help

- Use `python src/tests/run_tests.py --list` to see all available tests
- Check the verbose output with `--verbose` for more detailed error information
- Review test logs and error messages carefully

## Performance Considerations

- Tests should run quickly (< 1 second per test ideally)
- Use mocking for external dependencies
- Consider marking slow tests with appropriate decorators
- Run performance-critical tests separately if needed

## Future Enhancements

Planned improvements to the testing framework:

1. Parallel test execution for faster runs
2. Integration with more CI/CD platforms
3. Automated performance regression testing
4. Extended coverage for model training tests
5. Property-based testing for complex algorithms
