import os
import sys
import unittest

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnn_vuln_detection.naming_analysis.patterns import (
    BAD_NAMES,
    NAMING_PATTERNS,
    identify_naming_convention,
)


class TestNamingPatterns(unittest.TestCase):
    """Test cases for naming pattern recognition."""

    def test_snake_case_pattern(self) -> None:
        """Test snake_case pattern recognition."""
        snake_case_pattern = NAMING_PATTERNS["snake_case"]

        # Valid snake_case examples
        valid_names = [
            "hello",
            "hello_world",
            "get_user_name",
            "calculate_total_sum",
            "variable_name_123",
            "a_b_c_d",
            "single",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                assert snake_case_pattern.match(name), f"{name} should match snake_case"

        # Invalid snake_case examples
        invalid_names = [
            "HelloWorld",
            "helloWorld",
            "HELLO_WORLD",
            "hello-world",
            "hello.world",
            "_hello",
            "hello_",
            "hello__world",
            "Hello_World",
            "123_hello",
            "",
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                assert not snake_case_pattern.match(name), (
                    f"{name} should not match snake_case"
                )

    def test_camel_case_pattern(self) -> None:
        """Test camelCase pattern recognition."""
        camel_case_pattern = NAMING_PATTERNS["camelCase"]

        # Valid camelCase examples
        valid_names = [
            "helloWorld",
            "getUserName",
            "calculateTotalSum",
            "variableName123",
            "aBcD",
            "getX",
            "isValid",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                assert camel_case_pattern.match(name), f"{name} should match camelCase"

        # Invalid camelCase examples
        invalid_names = [
            "HelloWorld",  # PascalCase
            "hello_world",  # snake_case
            "HELLO_WORLD",  # SCREAMING_SNAKE
            "hello-world",  # kebab-case
            "hello",  # single word lowercase
            "helloworld",  # single word lowercase
            "Hello",  # single word PascalCase
            "_hello",
            "hello_",
            "",
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                assert not camel_case_pattern.match(name), (
                    f"{name} should not match camelCase"
                )

    def test_pascal_case_pattern(self) -> None:
        """Test PascalCase pattern recognition."""
        pascal_case_pattern = NAMING_PATTERNS["PascalCase"]

        # Valid PascalCase examples
        valid_names = [
            "HelloWorld",
            "GetUserName",
            "CalculateTotalSum",
            "VariableName123",
            "Person",
            "MyClass",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                assert pascal_case_pattern.match(name), (
                    f"{name} should match PascalCase"
                )

        # Invalid PascalCase examples
        invalid_names = [
            "helloWorld",  # camelCase
            "hello_world",  # snake_case
            "HELLO_WORLD",  # SCREAMING_SNAKE
            "hello-world",  # kebab-case
            "hello",  # single word lowercase
            "HELLO",  # single word uppercase
            "_Hello",
            "Hello_",
            "H",  # single character
            "",
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                assert not pascal_case_pattern.match(name), (
                    f"{name} should not match PascalCase"
                )

    def test_screaming_snake_pattern(self) -> None:
        """Test SCREAMING_SNAKE pattern recognition."""
        screaming_snake_pattern = NAMING_PATTERNS["SCREAMING_SNAKE"]

        # Valid SCREAMING_SNAKE examples
        valid_names = [
            "HELLO",
            "HELLO_WORLD",
            "GET_USER_NAME",
            "CALCULATE_TOTAL_SUM",
            "VARIABLE_NAME_123",
            "A_B_C_D",
            "CONSTANT_VALUE",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                assert screaming_snake_pattern.match(name), (
                    f"{name} should match SCREAMING_SNAKE"
                )

        # Invalid SCREAMING_SNAKE examples
        invalid_names = [
            "HelloWorld",  # PascalCase
            "helloWorld",  # camelCase
            "hello_world",  # snake_case
            "hello-world",  # kebab-case
            "hello.world",  # dot notation
            "_HELLO",
            "HELLO_",
            "HELLO__WORLD",
            "Hello_World",  # mixed case
            "",
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                assert not screaming_snake_pattern.match(name), (
                    f"{name} should not match SCREAMING_SNAKE"
                )

    def test_kebab_case_pattern(self) -> None:
        """Test kebab-case pattern recognition."""
        kebab_case_pattern = NAMING_PATTERNS["kebab-case"]

        # Valid kebab-case examples
        valid_names = [
            "hello",
            "hello-world",
            "get-user-name",
            "calculate-total-sum",
            "variable-name-123",
            "a-b-c-d",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                assert kebab_case_pattern.match(name), f"{name} should match kebab-case"

        # Invalid kebab-case examples
        invalid_names = [
            "HelloWorld",  # PascalCase
            "helloWorld",  # camelCase
            "hello_world",  # snake_case
            "HELLO_WORLD",  # SCREAMING_SNAKE
            "hello.world",  # dot notation
            "-hello",
            "hello-",
            "hello--world",
            "Hello-World",  # mixed case
            "123-hello",
            "",
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                assert not kebab_case_pattern.match(name), (
                    f"{name} should not match kebab-case"
                )

    def test_dot_case_pattern(self) -> None:
        """Test lower.dot.case pattern recognition."""
        dot_case_pattern = NAMING_PATTERNS["lower.dot.case"]

        # Valid lower.dot.case examples
        valid_names = [
            "hello",
            "hello.world",
            "get.user.name",
            "calculate.total.sum",
            "variable.name.123",
            "a.b.c.d",
        ]

        for name in valid_names:
            with self.subTest(name=name):
                assert dot_case_pattern.match(name), (
                    f"{name} should match lower.dot.case"
                )

        # Invalid lower.dot.case examples
        invalid_names = [
            "HelloWorld",  # PascalCase
            "helloWorld",  # camelCase
            "hello_world",  # snake_case
            "HELLO_WORLD",  # SCREAMING_SNAKE
            "hello-world",  # kebab-case
            ".hello",
            "hello.",
            "hello..world",
            "Hello.World",  # mixed case
            "123.hello",
            "",
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                assert not dot_case_pattern.match(name), (
                    f"{name} should not match lower.dot.case"
                )

    def test_identify_naming_convention_function(self) -> None:
        """Test the identify_naming_convention function."""
        test_cases = [
            # snake_case
            ("hello_world", "snake_case"),
            ("get_user_name", "snake_case"),
            ("variable_123", "snake_case"),
            # camelCase
            ("helloWorld", "camelCase"),
            ("getUserName", "camelCase"),
            ("variableName", "camelCase"),
            ("variable123Name", "camelCase"),
            ("variableName1234", "camelCase"),
            # PascalCase
            ("HelloWorld", "PascalCase"),
            ("GetUserName", "PascalCase"),
            ("MyClass", "PascalCase"),
            # SCREAMING_SNAKE
            ("HELLO_WORLD", "SCREAMING_SNAKE"),
            ("GET_USER_NAME", "SCREAMING_SNAKE"),
            ("CONSTANT_VALUE", "SCREAMING_SNAKE"),
            # kebab-case
            ("hello-world", "kebab-case"),
            ("get-user-name", "kebab-case"),
            # lower.dot.case
            ("hello.world", "lower.dot.case"),
            ("get.user.name", "lower.dot.case"),
            # unknown
            ("123invalid", "unknown"),
            ("_invalid", "unknown"),
            ("invalid_", "unknown"),
            ("", "unknown"),
            ("MiXeD_cAsE", "unknown"),
        ]

        for name, expected_convention in test_cases:
            with self.subTest(name=name, expected=expected_convention):
                result = identify_naming_convention(name)
                assert result == expected_convention, (
                    f"Expected {expected_convention} for '{name}', got {result}"
                )

    def test_bad_names_set(self) -> None:
        """Test that BAD_NAMES contains expected bad variable names."""
        expected_bad_names = {
            "temp",
            "asdf",
            "foo",
            "bar",
            "baz",
            "data",
            "value",
            "thing",
            "stuff",
            "var",
            "obj",
            "x",
            "y",
            "z",
        }

        # Check that all expected bad names are present
        for bad_name in expected_bad_names:
            assert bad_name in BAD_NAMES, f"'{bad_name}' should be in BAD_NAMES"

        # Check that BAD_NAMES is a set
        assert isinstance(BAD_NAMES, set)

        # Check that BAD_NAMES has the expected minimum size
        assert len(BAD_NAMES) >= len(expected_bad_names)

    def test_naming_patterns_dict_structure(self) -> None:
        """Test that NAMING_PATTERNS has the expected structure."""
        expected_patterns = {
            "snake_case",
            "camelCase",
            "PascalCase",
            "SCREAMING_SNAKE",
            "kebab-case",
            "lower.dot.case",
        }

        # Check that all expected patterns are present
        for pattern_name in expected_patterns:
            assert pattern_name in NAMING_PATTERNS, (
                f"'{pattern_name}' should be in NAMING_PATTERNS"
            )

        # Check that NAMING_PATTERNS is a dictionary
        assert isinstance(NAMING_PATTERNS, dict)

        # Check that all values are compiled regex patterns
        import re

        for pattern_name, pattern in NAMING_PATTERNS.items():
            assert isinstance(pattern, re.Pattern), (
                f"Pattern for '{pattern_name}' should be a compiled regex"
            )

    def test_edge_cases(self) -> None:
        """Test edge cases for naming convention identification."""
        edge_cases = [
            # Single characters
            ("a", "snake_case"),  # Single lowercase letter
            (
                "A",
                "SCREAMING_SNAKE",
            ),  # Single uppercase letter matches SCREAMING_SNAKE first
            # Numbers with patterns that actually work
            ("abc", "snake_case"),  # Multiple letters
            ("ABC", "SCREAMING_SNAKE"),  # Multiple uppercase letters
            # Very long names
            ("very_long_variable_name_with_many_parts", "snake_case"),
            ("veryLongVariableNameWithManyParts", "camelCase"),
            ("VeryLongVariableNameWithManyParts", "PascalCase"),
            # Mixed patterns that should be unknown
            ("snake_CaseWrong", "unknown"),
            ("camelCase_wrong", "unknown"),
            ("PascalCase_wrong", "unknown"),
        ]

        for name, expected_convention in edge_cases:
            with self.subTest(name=name, expected=expected_convention):
                result = identify_naming_convention(name)
                assert result == expected_convention, (
                    f"Expected {expected_convention} for '{name}', got {result}"
                )


if __name__ == "__main__":
    unittest.main()
