#!/bin/bash
# Continuous Integration Test Script
# This script runs all tests and checks for the project

set -e  # Exit on any error

echo "🏗️  GNN Vulnerability Detection - CI Test Suite"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python3 --version

# Check if we're in the right directory
# This script should be run from the project root, not from scripts/
if [[ ! -f "pyproject.toml" ]]; then
    # If we're in scripts/, try to cd to parent directory
    if [[ -f "../pyproject.toml" ]]; then
        print_status "Moving to project root directory..."
        cd ..
    else
        print_error "Not in the project root directory. Please run from the project root."
        print_error "Usage: ./scripts/ci_test.sh (from project root) or cd to project root and run ./scripts/ci_test.sh"
        exit 1
    fi
fi

# Install dependencies if requirements.txt exists
if [[ -f "requirements.txt" ]]; then
    print_status "Installing dependencies..."
    python3 -m pip install -r requirements.txt
else
    print_warning "No requirements.txt found. Proceeding without installing dependencies."
fi

# Run syntax check
print_status "Running syntax check..."
if python3 -m py_compile src/gnn_vuln_detection/**/*.py; then
    print_success "Syntax check passed"
else
    print_error "Syntax check failed"
    exit 1
fi

# Run import tests
print_status "Testing imports..."
if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from gnn_vuln_detection.naming_analysis.patterns import identify_naming_convention
    from gnn_vuln_detection.code_representation.ast_parser import ASTParser
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"; then
    print_success "Import tests passed"
else
    print_error "Import tests failed"
    exit 1
fi

# Run main test suite
print_status "Running main test suite..."
if python3 src/tests/run_tests.py; then
    print_success "All tests passed!"
else
    print_error "Some tests failed"
    exit 1
fi

# Run specific component tests
print_status "Running AST parser tests..."
python3 src/tests/run_tests.py --module ast_parser

print_status "Running naming convention tests..."
python3 src/tests/run_tests.py --module naming_patterns
python3 src/tests/run_tests.py --module naming_analyzer

# Check for coverage if available
if command -v coverage &> /dev/null; then
    print_status "Running coverage analysis..."
    python3 src/tests/run_tests.py --coverage
else
    print_warning "Coverage not available. Install with: pip install coverage"
fi

# Final summary
echo ""
echo "CI Test Suite Completed Successfully"
echo "========================================"
print_success "All checks passed"
print_status "Project is ready for deployment/development"
