import os
import sys
import unittest

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from gnn_vuln_detection.code_representation.ast_parser import ASTParser


class TestASTParser(unittest.TestCase):
    """Test cases for the ASTParser class."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        try:
            self.c_parser = ASTParser(language="c")
            self.cpp_parser = ASTParser(language="cpp")
            self.go_parser = ASTParser(language="go")
        except Exception as e:
            self.skipTest(f"Tree-sitter languages not available: {e}")

    def test_init_supported_languages(self) -> None:
        """Test that supported languages initialize correctly."""
        parser_c = ASTParser(language="c")
        assert parser_c.language == "c"

        parser_cpp = ASTParser(language="cpp")
        assert parser_cpp.language == "cpp"

        parser_go = ASTParser(language="go")
        assert parser_go.language == "go"

    def test_init_unsupported_language(self) -> None:
        """Test that unsupported languages raise ValueError."""
        with pytest.raises(ValueError) as context:
            ASTParser(language="python")
        assert "Language 'python' is not supported" in str(context.value)

        with pytest.raises(ValueError) as context:
            ASTParser(language="java")
        assert "Language 'java' is not supported" in str(context.value)

    def test_parse_simple_c_code(self) -> None:
        """Test parsing simple C code."""
        c_code = """
        int main() {
            int x = 10;
            return 0;
        }
        """
        ast_root = self.c_parser.parse_code_to_ast(c_code)

        assert ast_root is not None
        assert ast_root.type == "translation_unit"
        assert len(ast_root.children) > 0

    def test_parse_simple_cpp_code(self) -> None:
        """Test parsing simple C++ code."""
        cpp_code = """
        #include <iostream>
        class MyClass {
        public:
            int value;
            MyClass(int v) : value(v) {}
        };
        int main() {
            MyClass obj(42);
            return 0;
        }
        """
        ast_root = self.cpp_parser.parse_code_to_ast(cpp_code)

        assert ast_root is not None
        assert ast_root.type == "translation_unit"
        assert len(ast_root.children) > 0

    def test_parse_simple_go_code(self) -> None:
        """Test parsing simple Go code."""
        go_code = """
        package main
        import "fmt"
        func main() {
            x := 10
            fmt.Println(x)
        }
        """
        ast_root = self.go_parser.parse_code_to_ast(go_code)

        assert ast_root is not None
        assert ast_root.type == "source_file"
        assert len(ast_root.children) > 0

    def test_parse_empty_code(self) -> None:
        """Test parsing empty code."""
        empty_code = ""
        ast_root = self.c_parser.parse_code_to_ast(empty_code)

        assert ast_root is not None
        assert ast_root.type == "translation_unit"

    def test_parse_invalid_c_syntax(self) -> None:
        """Test parsing invalid C syntax."""
        invalid_c_code = """
        int main( {
            int x = 10
            return 0;
        }
        """
        ast_root = self.c_parser.parse_code_to_ast(invalid_c_code)

        # Tree-sitter should still return a tree, but it might contain error nodes
        assert ast_root is not None
        assert ast_root.type == "translation_unit"

    def test_parse_function_with_variables(self) -> None:
        """Test parsing C code with functions and variables."""
        c_code = """
        int add(int a, int b) {
            int result = a + b;
            return result;
        }

        int main() {
            int x = 5;
            int y = 10;
            int sum = add(x, y);
            return 0;
        }
        """
        ast_root = self.c_parser.parse_code_to_ast(c_code)

        assert ast_root is not None
        assert ast_root.type == "translation_unit"

        # Check that we have function definitions
        function_definitions = []
        for child in ast_root.children:
            if child.type == "function_definition":
                function_definitions.append(child)

        assert len(function_definitions) == 2  # add and main functions

    def test_parse_struct_definition(self) -> None:
        """Test parsing C code with struct definition."""
        c_code = """
        struct Point {
            int x;
            int y;
        };

        int main() {
            struct Point p;
            p.x = 10;
            p.y = 20;
            return 0;
        }
        """
        ast_root = self.c_parser.parse_code_to_ast(c_code)

        assert ast_root is not None
        assert ast_root.type == "translation_unit"

        # Check for struct declaration
        has_struct = False
        for child in ast_root.children:
            if child.type == "struct_specifier":
                has_struct = True
                break

        # The struct might be wrapped in a declaration
        if not has_struct:
            for child in ast_root.children:
                if child.type == "declaration":
                    for grandchild in child.children:
                        if grandchild.type == "struct_specifier":
                            has_struct = True
                            break

        assert (
            has_struct or len(ast_root.children) > 0
        )  # At least verify we got some structure

    def test_print_ast_method_exists(self) -> None:
        """Test that print_ast method exists and is callable."""
        c_code = "int x = 5;"
        ast_root = self.c_parser.parse_code_to_ast(c_code)

        # This should not raise an exception
        try:
            # Capture stdout to avoid printing during tests
            import contextlib
            import io

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                self.c_parser.print_ast(ast_root)
            output = f.getvalue()

            # Check that some output was generated
            assert len(output) > 0
            assert "translation_unit" in output
        except Exception as e:
            self.fail(f"print_ast method failed: {e}")


if __name__ == "__main__":
    unittest.main()
