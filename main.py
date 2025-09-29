from collections import defaultdict

from src.gnn_vuln_detection.code_representation.ast_parser import ASTParser
from src.gnn_vuln_detection.naming_analysis.analyzer import (
    extract_identifiers,
    is_meaningful,
)
from src.gnn_vuln_detection.naming_analysis.patterns import (
    BAD_NAMES,
    identify_naming_convention,
)


def analyze_code(language="c", source_code=""):
    c_parser = ASTParser(language=language)
    if not source_code:
        source_code = """
        int my_function(int param1, int param2) {
            int result = param1 + param2;
            if (result > 10) {
                printf("Result is greater than 10\\n");
            } else {
                printf("Result is 10 or less\\n");
            }
            return result;
        }

        int main() {
            int x = 10;
            my_function(x, 5);
            for (int i = 0; i < 10; i++) {
                x += i;
            }
            if (x > 5) {
                printf("Hello\\n");
            }
            return 0;
        }
        """
    # c_ast_root = c_parser.parse_code_to_ast(source_code)

    identifiers = extract_identifiers(
        c_parser.parser.parse(bytes(source_code, "utf-8")),
        bytes(source_code, "utf8"),
        language,
    )
    findings = defaultdict(list)

    for name in set(identifiers):
        style = identify_naming_convention(name)
        if name.lower() in BAD_NAMES or not is_meaningful(name):
            findings["non_descriptive"].append(name)
        if language == "python" and style != "snake_case":
            findings["bad_python_style"].append(name)
        elif language in {"java", "typescript"} and style not in {
            "camelCase",
            "PascalCase",
        }:
            findings["bad_java_style"].append(name)
        if style == "unknown" and style not in {
            "snake_case",
            "camelCase",
            "PascalCase",
            "SCREAMING_SNAKE",
        }:
            findings["unknown_style"].append(name)
    return findings


if __name__ == "__main__":
    findings = analyze_code()
    print("Findings:")
    for category, names in findings.items():
        print(f"{category}: {', '.join(names) if names else 'None'}")
    # # Example usage of the AST parser
    # try:
    #     c_parser = ASTParser(language="c")
    #     c_code = """
    #     int main() {
    #         int x = 10;
    #         if (x > 5) {
    #             printf("Hello\\n");
    #         }
    #         return 0;
    #     }
    #     """
    #     c_ast_root = c_parser.parse_code_to_ast(c_code)
    #     print("--- C AST ---")
    #     c_parser.print_ast(c_ast_root)

    # except Exception as e:
    #     print(f"Błąd inicjalizacji lub parsowania C: {e}")
    #     print(
    #         "Upewnij się, że biblioteki tree-sitter dla C są poprawnie skonfigurowane i dostępne."
    #     )

    # graph_builder = GraphBuilder()
    # c_graph = graph_builder.build_graph(c_ast_root)
    # print("--- C Graph ---")
    # stats = graph_builder.get_graph_statistics(c_graph)
    # print(stats)
    # print(f"Graph: {c_graph}")
    # print("Nodes:", c_graph.number_of_nodes())
    # print("Edges:", c_graph.number_of_edges())

    # # Display the graph using networkx and matplotlib
    # plt.figure(figsize=(12, 8))
    # pos = nx.spring_layout(c_graph, k=1, iterations=50)
    # nx.draw(c_graph, pos, with_labels=True, node_color='lightblue',
    #     node_size=500, font_size=8, font_weight='bold',
    #     arrows=True, edge_color='gray')
    # plt.title("AST Graph Visualization")
    # plt.show()
