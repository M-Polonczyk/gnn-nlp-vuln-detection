# import javalang
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_go
import tree_sitter_java
from tree_sitter import Language, Parser

C_LANGUAGE = Language(tree_sitter_c.language())
CPP_LANGUAGE = Language(tree_sitter_cpp.language())
GO_LANGUAGE = Language(tree_sitter_go.language())
JAVA_LANGUAGE = Language(tree_sitter_java.language())


class ASTParser:
    def __init__(self, language: str = "c") -> None:
        """Initialize the AST parser for a specific programming language.

        Args:
            language (str, optional): The programming language to parse. Defaults to "c".

        Raises:
            ValueError: If the language is not supported.
        """
        self.parser = Parser()
        match language.lower():
            case "c":
                self.parser.language = C_LANGUAGE
            case "cpp":
                self.parser.language = CPP_LANGUAGE
            case "go":
                self.parser.language = GO_LANGUAGE
            case "java":
                self.parser.language = JAVA_LANGUAGE
            case _:
                msg = f"Language '{language}' is not supported."
                raise ValueError(msg)
        self.language = language

    def parse_code_to_ast(self, code_string: str):
        """Parses the given code string into an AST.

        Args:
            code_string (str): Source code as a string.

        Returns:
            Node: The root of the AST (tree-sitter object).
        """
        tree = self.parser.parse(bytes(code_string, "utf8"))
        # For Java, we use javalang
        # if self.language == "java":
        #     tree = javalang.parse.parse(code_string)
        #     return tree
        return tree.root_node

    def print_ast(self, node, indent="") -> None:
        """Helper function to print the AST structure (for debugging purposes)."""
        print(
            f"{indent}{node.type} [{node.start_point} - {node.end_point}] '{node.text.decode('utf8')}'",
        )
        for child in node.children:
            self.print_ast(child, indent + "  ")


if __name__ == "__main__":
    from textwrap import dedent

    try:
        c_parser = ASTParser(language="c")
        c_code = dedent(
            """
        int main() {
            int x = 10;
            if (x > 5) {
                printf("Hello\\n");
            }
            return 0;
        }
        """,
        )
        c_ast_root = c_parser.parse_code_to_ast(c_code)
        print("--- C AST ---")
        c_parser.print_ast(c_ast_root)

    except Exception as e:
        print(f"Błąd inicjalizacji lub parsowania C: {e}")

    try:
        cpp_parser = ASTParser(language="cpp")
        cpp_code = dedent(
            """
        #include <iostream>
        class MyClass {
        public:
            int value;
            MyClass(int v) : value(v) {}
            void printValue() { std::cout << value << std::endl; }
        };
        int main() {
            MyClass obj(42);
            obj.printValue();
            return 0;
        }
        """,
        )
        cpp_ast_root = cpp_parser.parse_code_to_ast(cpp_code)
        # print("\n--- C++ AST ---")
        # cpp_parser.print_ast(cpp_ast_root)

    except Exception as e:
        print(f"Błąd inicjalizacji lub parsowania C++: {e}")

    try:
        go_parser = ASTParser(language="go")
        go_code = dedent(
            """
        package main
        import "fmt"
        func main() {
            x := 10
            if x > 5 {
                fmt.Println("Hello")
            }
        }
        """,
        )
        go_ast_root = go_parser.parse_code_to_ast(go_code)
        # print("\n--- Go AST ---")
        # go_parser.print_ast(go_ast_root)
    except Exception as e:
        print(f"Błąd inicjalizacji lub parsowania Go: {e}")
