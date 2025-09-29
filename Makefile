PYTHON := python3

TEST_DIR := src/tests
SRC_DIR := src
SCRIPTS_DIR := scripts

.PHONY: help
help:
	@echo "GNN Vulnerability Detection - Testing Commands"
	@echo "=============================================="
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all tests"
	@echo "  test-ast          Run AST parser tests"
	@echo "  test-naming       Run naming convention tests"
	@echo "  test-verbose      Run tests with verbose output"
	@echo "  test-quiet        Run tests with minimal output"
	@echo "  test-coverage     Run tests with coverage analysis"
	@echo "  test-list         List all available tests"
	@echo "  test-verify       Run verification tests"
	@echo "  test-ci           Run full CI test suite"
	@echo ""
	@echo "Development:"
	@echo "  install-dev       Install development dependencies"
	@echo "  clean             Clean up generated files"
	@echo "  format            Format code (if black is installed)"
	@echo "  lint              Run linting (if flake8 is installed)"

# Test targets
.PHONY: test
test:
	@echo "🚀 Running all tests..."
	@$(PYTHON) $(TEST_DIR)/run_tests.py

.PHONY: test-ast
test-ast:
	@echo "🧠 Running AST parser tests..."
	@$(PYTHON) $(TEST_DIR)/run_tests.py --module ast_parser

.PHONY: test-naming
test-naming:
	@echo "🏷️  Running naming convention tests..."
	@$(PYTHON) $(TEST_DIR)/run_tests.py --module naming_patterns
	@$(PYTHON) $(TEST_DIR)/run_tests.py --module naming_analyzer

.PHONY: test-verbose
test-verbose:
	@echo "🔍 Running tests with verbose output..."
	@$(PYTHON) $(TEST_DIR)/run_tests.py --verbose --verbose

.PHONY: test-quiet
test-quiet:
	@echo "🤫 Running tests quietly..."
	@$(PYTHON) $(TEST_DIR)/run_tests.py --quiet

.PHONY: test-coverage
test-coverage:
	@echo "📊 Running tests with coverage analysis..."
	@$(PYTHON) $(TEST_DIR)/run_tests.py --coverage

.PHONY: test-list
test-list:
	@echo "📋 Listing available tests..."
	@$(PYTHON) $(TEST_DIR)/run_tests.py --list

.PHONY: test-ci
test-ci:
	@echo "🚀 Running full CI test suite..."
	@bash $(SCRIPTS_DIR)/ci_test.sh

.PHONY: clean
clean:
	@echo "🧹 Cleaning up generated files..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name ".coverage" -delete
	@rm -rf coverage_html/ 2>/dev/null || true
	@rm -rf .pytest_cache/ 2>/dev/null || true
	@rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

.PHONY: format
format:
	@echo "🎨 Formatting code..."
	@command -v ruff >/dev/null 2>&1 && ruff format $(SRC_DIR)

.PHONY: lint
lint:
	@echo "🔍 Running linting..."
	@command -v ruff >/dev/null 2>&1 && ruff check --fix $(SRC_DIR)

.PHONY: t
t: test

.PHONY: tv
tv: test-verbose

.PHONY: tq
tq: test-quiet

.PHONY: tc
tc: test-coverage
