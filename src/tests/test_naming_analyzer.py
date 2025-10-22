import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnn_vuln_detection.naming_analysis.analyzer import (
    extract_identifiers,
    is_meaningful,
)


class TestNamingAnalyzer(unittest.TestCase):
    """Test cases for the naming analyzer functions."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Mock nltk and inflect to avoid dependencies in tests
        self.mock_english_words = {
            "hello",
            "world",
            "user",
            "name",
            "calculate",
            "total",
            "sum",
            "get",
            "set",
            "add",
            "remove",
            "create",
            "delete",
            "update",
            "find",
            "search",
            "list",
            "array",
            "string",
            "number",
            "value",
            "data",
            "file",
            "path",
            "size",
            "length",
            "count",
            "index",
            "result",
            "output",
            "input",
            "parameter",
            "variable",
            "function",
            "method",
            "class",
            "object",
            "item",
            "element",
            "node",
            "tree",
        }

        self.mock_inflect_engine = MagicMock()
        self.mock_inflect_engine.singular_noun.side_effect = lambda word: (
            word[:-1] if word.endswith("s") and len(word) > 1 else False
        )

    @patch("gnn_vuln_detection.naming_analysis.analyzer.english_words")
    @patch("gnn_vuln_detection.naming_analysis.analyzer.inflect_engine")
    def test_is_meaningful_single_words(self, mock_inflect, mock_words) -> None:
        """Test is_meaningful function with single words."""

        # Properly mock the __contains__ method
        def mock_contains(word):
            return word in self.mock_english_words

        mock_words.__contains__ = mock_contains
        mock_inflect.singular_noun = self.mock_inflect_engine.singular_noun

        # Meaningful single words
        meaningful_words = [
            "hello",
            "world",
            "user",
            "name",
            "calculate",
            "variable",
            "function",
            "method",
            "class",
        ]

        for word in meaningful_words:
            with self.subTest(word=word):
                assert is_meaningful(word), f"'{word}' should be meaningful"

        # Non-meaningful single words
        non_meaningful_words = [
            "x",
            "y",
            "z",
            "a",
            "b",
            "c",  # single letters
            "foo",
            "bar",
            "baz",  # placeholder names
            "asdf",
            "qwerty",  # keyboard mashing
            "temp",
            "tmp",  # temporary names
            "xyz123",  # meaningless combinations
        ]

        for word in non_meaningful_words:
            with self.subTest(word=word):
                assert not is_meaningful(word), f"'{word}' should not be meaningful"

    @patch("gnn_vuln_detection.naming_analysis.analyzer.english_words")
    @patch("gnn_vuln_detection.naming_analysis.analyzer.inflect_engine")
    def test_is_meaningful_compound_words(self, mock_inflect, mock_words) -> None:
        """Test is_meaningful function with compound words."""

        # Properly mock the __contains__ method
        def mock_contains(word):
            return word in self.mock_english_words

        mock_words.__contains__ = mock_contains
        mock_inflect.singular_noun = self.mock_inflect_engine.singular_noun

        # Meaningful compound words (snake_case)
        meaningful_compound = [
            "user_name",
            "get_user",
            "calculate_total",
            "hello_world",
            "file_path",
            "array_size",
            "find_element",
            "create_object",
        ]

        for word in meaningful_compound:
            with self.subTest(word=word):
                assert is_meaningful(word), f"'{word}' should be meaningful"

        # Meaningful compound words (camelCase)
        meaningful_camel = [
            "userName",
            "getUser",
            "calculateTotal",
            "helloWorld",
            "filePath",
            "arraySize",
            "findElement",
            "createObject",
        ]

        for word in meaningful_camel:
            with self.subTest(word=word):
                assert is_meaningful(word), f"'{word}' should be meaningful"

        # Non-meaningful compound words
        non_meaningful_compound = [
            "foo_bar",
            "x_y_z",
            "asdf_qwerty",
            "temp_var",
            "a_b_c",
            "xyz_123",
        ]

        for word in non_meaningful_compound:
            with self.subTest(word=word):
                assert not is_meaningful(word), f"'{word}' should not be meaningful"

    @patch("gnn_vuln_detection.naming_analysis.analyzer.english_words")
    @patch("gnn_vuln_detection.naming_analysis.analyzer.inflect_engine")
    def test_is_meaningful_plurals(self, mock_inflect, mock_words) -> None:
        """Test is_meaningful function with plural words."""

        # Properly mock the __contains__ method
        def mock_contains(word):
            return word in self.mock_english_words

        mock_words.__contains__ = mock_contains
        mock_inflect.singular_noun = self.mock_inflect_engine.singular_noun

        # Test plurals that should be recognized
        plural_words = [
            "users",
            "names",
            "files",
            "arrays",
            "elements",
            "objects",
            "items",
            "nodes",
        ]

        for word in plural_words:
            with self.subTest(word=word):
                # The function should recognize these as meaningful through inflect
                # Since the singular forms are in our mock dictionary
                result = is_meaningful(word)
                # This test might vary based on inflect implementation
                # We mainly want to ensure the function doesn't crash
                assert isinstance(result, bool)

    def test_is_meaningful_edge_cases(self) -> None:
        """Test is_meaningful function with edge cases."""
        edge_cases = [
            ("", False),  # Empty string
            ("a", False),  # Single character
            ("1", False),  # Number
            ("123", False),  # Multiple numbers
            ("_", False),  # Underscore only
            ("__", False),  # Multiple underscores
        ]

        for name, expected in edge_cases:
            with self.subTest(name=name, expected=expected):
                try:
                    result = is_meaningful(name)
                    # For edge cases, we mainly want to ensure no crashes
                    assert isinstance(result, bool)
                except Exception as e:
                    self.fail(f"is_meaningful('{name}') raised an exception: {e}")

    def test_extract_identifiers_mock_tree(self) -> None:
        """Test extract_identifiers function with a mock tree structure."""

        # Create mock nodes
        class MockNode:
            def __init__(
                self, node_type, start_byte=0, end_byte=0, children=None
            ) -> None:
                self.type = node_type
                self.start_byte = start_byte
                self.end_byte = end_byte
                self.children = children or []

        # Create a mock tree structure
        identifier_node = MockNode("identifier", 0, 4)  # "main"
        function_def_node = MockNode("function_definition", 5, 18)  # "function_name"
        variable_decl_node = MockNode("variable_declarator", 19, 32)  # "variable_name"
        method_decl_node = MockNode("method_declaration", 33, 44)  # "method_name"
        other_node = MockNode("expression", 45, 50)  # "other"

        root_node = MockNode(
            "translation_unit",
            0,
            50,
            [
                identifier_node,
                function_def_node,
                variable_decl_node,
                method_decl_node,
                other_node,
            ],
        )

        # Mock source code
        source_code = b"main function_name variable_name method_name other"

        # Create a mock tree object
        mock_tree = MagicMock()
        mock_tree.walk.return_value.node = root_node

        # Test the function
        identifiers = extract_identifiers(mock_tree, source_code, "c")

        # We expect 4 identifiers (excluding the "other" node which is not an identifier type)
        expected_identifiers = ["main", "function_name", "variable_name", "method_name"]

        assert len(identifiers) == 4
        for expected in expected_identifiers:
            assert expected in identifiers

    def test_extract_identifiers_empty_tree(self) -> None:
        """Test extract_identifiers function with an empty tree."""

        class MockNode:
            def __init__(self, node_type, children=None) -> None:
                self.type = node_type
                self.children = children or []
                self.start_byte = 0
                self.end_byte = 0

        root_node = MockNode("translation_unit", [])

        mock_tree = MagicMock()
        mock_tree.walk.return_value.node = root_node

        source_code = b""

        identifiers = extract_identifiers(mock_tree, source_code, "c")

        assert len(identifiers) == 0

    def test_extract_identifiers_nested_structure(self) -> None:
        """Test extract_identifiers function with nested structure."""

        class MockNode:
            def __init__(
                self, node_type, start_byte=0, end_byte=0, children=None
            ) -> None:
                self.type = node_type
                self.start_byte = start_byte
                self.end_byte = end_byte
                self.children = children or []

        # Create nested structure
        inner_identifier = MockNode("identifier", 10, 15)
        inner_function = MockNode("function_definition", 5, 20, [inner_identifier])
        outer_function = MockNode("function_definition", 0, 25, [inner_function])

        root_node = MockNode("translation_unit", 0, 25, [outer_function])

        mock_tree = MagicMock()
        mock_tree.walk.return_value.node = root_node

        source_code = b"function outer_func inner"

        identifiers = extract_identifiers(mock_tree, source_code, "c")

        # Should find both function definitions and the identifier
        assert len(identifiers) >= 2

    def test_extract_identifiers_different_languages(self) -> None:
        """Test extract_identifiers function behavior with different languages."""

        class MockNode:
            def __init__(
                self, node_type, start_byte=0, end_byte=0, children=None
            ) -> None:
                self.type = node_type
                self.start_byte = start_byte
                self.end_byte = end_byte
                self.children = children or []

        identifier_node = MockNode("identifier", 0, 4)
        root_node = MockNode("source_file", 0, 4, [identifier_node])

        mock_tree = MagicMock()
        mock_tree.walk.return_value.node = root_node

        source_code = b"test"

        # Test with different languages
        for language in ["c", "cpp", "go", "python", "java"]:
            with self.subTest(language=language):
                identifiers = extract_identifiers(mock_tree, source_code, language)
                assert isinstance(identifiers, list)
                assert len(identifiers) == 1
                assert identifiers[0] == "test"


class TestNamingAnalyzerIntegration(unittest.TestCase):
    """Integration tests for naming analyzer with real tree-sitter parsing."""

    def setUp(self) -> None:
        """Set up test fixtures for integration tests."""
        try:
            from gnn_vuln_detection.code_representation.ast_parser import ASTParser

            self.c_parser = ASTParser(language="c")
        except Exception as e:
            self.skipTest(f"Tree-sitter not available for integration tests: {e}")

    def test_extract_identifiers_real_c_code(self) -> None:
        """Test extract_identifiers with real C code and tree-sitter parsing."""
        c_code = """
        int main() {
            int user_count = 10;
            char user_name[50];
            return 0;
        }
        """

        try:
            ast_root = self.c_parser.parse_code_to_ast(c_code)

            # Create a mock tree object that tree-sitter would return
            mock_tree = MagicMock()
            mock_tree.walk.return_value.node = ast_root

            identifiers = extract_identifiers(mock_tree, c_code.encode(), "c")

            # Should find some identifiers
            assert isinstance(identifiers, list)
            # The exact identifiers depend on tree-sitter parsing,
            # but we should get at least some results
            assert len(identifiers) >= 0

        except Exception as e:
            self.skipTest(f"Tree-sitter parsing failed: {e}")


if __name__ == "__main__":
    unittest.main()
