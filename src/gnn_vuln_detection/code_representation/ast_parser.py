# import javalang
from dataclasses import dataclass

import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_go
import tree_sitter_java
from tree_sitter import Language, Node, Parser

C_LANGUAGE = Language(tree_sitter_c.language())
CPP_LANGUAGE = Language(tree_sitter_cpp.language())
GO_LANGUAGE = Language(tree_sitter_go.language())
JAVA_LANGUAGE = Language(tree_sitter_java.language())


@dataclass
class Code:
    code_string: str
    ast_root: Node


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

    def parse_code_to_ast(self, code: str | bytes) -> Node:
        """Parses the given code string into an AST.

        Args:
            code_string (str): Source code as a string.

        Returns:
            Node: The root of the AST (tree-sitter object).
        """
        if isinstance(code, str):
            code = bytes(code, encoding="utf8")
        tree = self.parser.parse(code, encoding="utf8")
        # For Java, we use javalang
        # if self.language == "java":
        #     tree = javalang.parse.parse(code_string)
        #     return tree
        return tree.root_node

    def print_ast(self, node: Node, indent: str = "") -> None:
        """Helper function to print the AST structure (for debugging purposes)."""
        print(  # noqa: T201
            f"{indent}{node.type} [{node.start_point} - {node.end_point}] \
                '{node.text.decode('utf8') if node.text else ''}'",
        )
        for child in node.children:
            self.print_ast(child, indent + "  ")

    def ast_to_code(self, node: Node, source_bytes: bytes) -> str:
        """
        Reconstruct source code for a node using original source bytes.
        """
        return source_bytes[node.start_byte : node.end_byte].decode("utf8")

    def remove_comments_from_node(self, node: Node) -> Node | None:
        """
        Removes comments from the given Tree-sitter AST node.
        """
        raise NotImplementedError("Comment removal not implemented yet.")
        source = node.text
        if not source:
            return ""
        comment_ranges: list[tuple[int, int]] = []

        def collect(n: Node) -> None:
            if n.type == "comment":
                comment_ranges.append((n.start_byte, n.end_byte))
            for child in n.children:
                collect(child)

        collect(node)

        # Remove comments from the end backward to preserve offsets
        for start, end in sorted(comment_ranges, reverse=True):
            source = source[:start] + source[end:]

        return source.decode("utf8")

    def _remove_comments(self, root: Node, source_bytes: bytes) -> bytes:
        """
        Removes all comment nodes using absolute Tree-sitter byte offsets.
        """

        comment_ranges: list[tuple[int, int]] = []

        def collect(n: Node) -> None:
            if n.type == "comment":
                comment_ranges.append((n.start_byte, n.end_byte))
            for child in n.children:
                collect(child)

        collect(root)

        for start, end in sorted(comment_ranges, reverse=True):
            source_bytes = source_bytes[:start] + source_bytes[end:]

        return source_bytes

    def remove_comments(self, code: Code | str) -> str:
        """Removes comments from the given code string.

        Args:
            code_string (str): Source code as a string.

        Returns:
            str: Code string with comments removed.
        """
        if isinstance(code, str):
            ast_root = self.parse_code_to_ast(code)
            return self._remove_comments(ast_root, bytes(code, encoding="utf8")).decode(
                "utf8"
            )
        if isinstance(code, Code):
            return self._remove_comments(
                code.ast_root, bytes(code.code_string, encoding="utf8")
            ).decode("utf8")
        msg = "Input must be a code string or Code object."
        raise TypeError(msg)

    def cleanup_code(self, code: str) -> str:
        """Cleans up the code string by removing comments and unnecessary assertions."""
        # Additional cleanup steps can be added here if needed
        return self.remove_comments(code)


if __name__ == "__main__":
    from textwrap import dedent

    try:
        c_parser = ASTParser(language="c")
        c_code = dedent(
            """
        /* Multiline
         comment */

        int main() {
        // comment
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
        c_ast_root_no_comments = c_parser.parse_code_to_ast(
            c_parser.remove_comments(c_code)
        )
        print("\n--- C AST without comments ---")
        c_parser.print_ast(c_ast_root_no_comments)

    except Exception as e:
        print(f"Błąd inicjalizacji lub parsowania C: {e}")

    # try:
    #     cpp_parser = ASTParser(language="cpp")
    #     cpp_code = dedent(
    #         """
    #     #include <iostream>
    #     class MyClass {
    #     public:
    #         int value;
    #         MyClass(int v) : value(v) {}
    #         void printValue() { std::cout << value << std::endl; }
    #     };
    #     int main() {
    #         MyClass obj(42);
    #         obj.printValue();
    #         return 0;
    #     }
    #     """,
    #     )
    #     cpp_ast_root = cpp_parser.parse_code_to_ast(cpp_code)
    #     # print("\n--- C++ AST ---")
    #     # cpp_parser.print_ast(cpp_ast_root)

    # except Exception as e:
    #     print(f"Błąd inicjalizacji lub parsowania C++: {e}")

    # try:
    #     go_parser = ASTParser(language="go")
    #     go_code = dedent(
    #         """
    #     package main
    #     import "fmt"
    #     func main() {
    #         x := 10
    #         if x > 5 {
    #             fmt.Println("Hello")
    #         }
    #     }
    #     """,
    #     )
    #     go_ast_root = go_parser.parse_code_to_ast(go_code)
    #     # print("\n--- Go AST ---")
    #     # go_parser.print_ast(go_ast_root)
    # except Exception as e:
    #     print(f"Błąd inicjalizacji lub parsowania Go: {e}")
