#!/usr/bin/env python3
"""
Test runner for GNN Vulnerability Detection project.
Runs all tests for AST parsers and naming convention identifiers.
"""

import argparse
import os
import sys
import time
import unittest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_all_tests(verbosity=2, buffer=False):
    """Run all tests and return the test results."""
    print("🔍 Discovering tests...")

    # Discover and run all tests in the tests directory
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="test_*.py")

    test_count = suite.countTestCases()
    print(f"📊 Found {test_count} tests")

    # Run the tests
    print("🚀 Running tests...\n")
    start_time = time.time()

    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        buffer=buffer,
        stream=sys.stdout,
    )
    result = runner.run(suite)

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n⏱️  Tests completed in {duration:.2f} seconds")
    return result


def run_specific_test_module(module_name, verbosity=2):
    """Run tests for a specific module."""
    try:
        print(f"🎯 Running tests for module: {module_name}")

        # Import the specific test module
        test_module = __import__(f"test_{module_name}", fromlist=[""])

        # Create a test suite from the module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)

        test_count = suite.countTestCases()
        print(f"📊 Found {test_count} tests in module")

        # Run the tests
        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)

        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⏱️  Tests completed in {duration:.2f} seconds")

        return result
    except ImportError as e:
        print(f"❌ Error importing test module 'test_{module_name}': {e}")
        return None


def run_specific_test_class(module_name, class_name, verbosity=2):
    """Run tests for a specific test class."""
    try:
        print(f"🎯 Running tests for class: {class_name} in module: {module_name}")

        # Import the specific test module
        test_module = __import__(f"test_{module_name}", fromlist=[""])

        # Get the specific test class
        test_class = getattr(test_module, class_name)

        # Create a test suite from the class
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)

        test_count = suite.countTestCases()
        print(f"📊 Found {test_count} tests in class")

        # Run the tests
        start_time = time.time()
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)

        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⏱️  Tests completed in {duration:.2f} seconds")

        return result
    except (ImportError, AttributeError) as e:
        print(
            f"❌ Error running test class '{class_name}' in module 'test_{module_name}': {e}",
        )
        return None


def list_available_tests() -> None:
    """List all available test modules and their test classes."""
    print("📋 Available test modules and classes:\n")

    test_dir = os.path.dirname(__file__)
    test_files = [
        f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")
    ]

    for test_file in sorted(test_files):
        module_name = test_file[:-3]  # Remove .py extension
        print(f"📁 {module_name}")

        try:
            # Import the module to list its test classes
            test_module = __import__(module_name, fromlist=[""])

            # Find all test classes in the module
            test_classes = [
                name
                for name in dir(test_module)
                if name.startswith("Test")
                and hasattr(getattr(test_module, name), "__bases__")
            ]

            for test_class in test_classes:
                print(f"   🧪 {test_class}")

        except Exception as e:
            print(f"   ❌ Error loading module: {e}")

        print()


def run_coverage_analysis():
    """Run tests with coverage analysis if coverage is available."""
    try:
        import coverage

        print("📊 Running tests with coverage analysis...")

        # Start coverage
        cov = coverage.Coverage()
        cov.start()

        # Run tests
        result = run_all_tests(verbosity=1, buffer=True)

        # Stop coverage and save
        cov.stop()
        cov.save()

        print("\n📈 Coverage Report:")
        print("-" * 50)
        cov.report()

        # Generate HTML report
        html_dir = os.path.join(os.path.dirname(__file__), "..", "..", "coverage_html")
        cov.html_report(directory=html_dir)
        print(f"\n📄 HTML coverage report generated in: {html_dir}")

        return result

    except ImportError:
        print("⚠️  Coverage module not installed. Install with: pip install coverage")
        print("Running tests without coverage analysis...\n")
        return run_all_tests()


def create_test_parser():
    """Create argument parser for test runner."""
    parser = argparse.ArgumentParser(
        description="Run tests for GNN Vulnerability Detection project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                           # Run all tests
  python run_tests.py --module ast_parser       # Run specific module tests
  python run_tests.py --class TestASTParser     # Run specific test class
  python run_tests.py --list                    # List available tests
  python run_tests.py --coverage                # Run with coverage analysis
  python run_tests.py --quiet                   # Run with minimal output
        """,
    )

    parser.add_argument(
        "--module",
        "-m",
        help="Run tests for specific module (e.g., ast_parser, naming_patterns)",
    )

    parser.add_argument(
        "--class",
        "-c",
        dest="test_class",
        help="Run tests for specific test class (requires --module)",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available test modules and classes",
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage analysis",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=2,
        help="Increase verbosity (can be used multiple times)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Run tests with minimal output",
    )

    parser.add_argument(
        "--failfast",
        "-f",
        action="store_true",
        help="Stop on first failure",
    )

    return parser


def print_test_summary(result) -> None:
    """Print a summary of test results."""
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print(f"✅ All tests passed! Ran {result.testsRun} tests successfully.")
        if result.skipped:
            print(f"⏭️  Skipped {len(result.skipped)} tests.")
    else:
        failures = len(result.failures)
        errors = len(result.errors)
        total_issues = failures + errors

        print("❌ Tests completed with issues:")
        print(f"   • {failures} failures")
        print(f"   • {errors} errors")
        print(f"   • {result.testsRun - total_issues} passed")

        if result.failures:
            print("\n📋 Failed tests:")
            for test, _traceback in result.failures:
                print(f"   • {test}")

        if result.errors:
            print("\n📋 Error tests:")
            for test, _traceback in result.errors:
                print(f"   • {test}")

    print("=" * 60)


def main() -> None:
    """Main function to run tests based on command line arguments."""
    parser = create_test_parser()
    args = parser.parse_args()

    # Handle quiet mode
    verbosity = 0 if args.quiet else args.verbose

    # List available tests
    if args.list:
        list_available_tests()
        return

    # Determine which tests to run
    result = None
    if args.coverage:
        result = run_coverage_analysis()
    elif args.test_class:
        if not args.module:
            print("❌ Error: --class requires --module to be specified")
            sys.exit(1)
        result = run_specific_test_class(args.module, args.test_class, verbosity)
    elif args.module:
        result = run_specific_test_module(args.module, verbosity)
    else:
        result = run_all_tests(verbosity)

    if result is None:
        print("❌ Test execution failed")
        sys.exit(1)

    # Print summary and exit
    print_test_summary(result)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
