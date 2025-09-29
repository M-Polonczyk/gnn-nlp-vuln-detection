#!/usr/bin/env python3
"""Emaple usage of loading and analyzing C code samples."""

import sys
from pathlib import Path

import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from tree_sitter import Node

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.code_representation.ast_parser import ASTParser
from gnn_vuln_detection.code_representation.graph_builder import GraphBuilder
from gnn_vuln_detection.data_processing.graph_converter import ASTToGraphConverter
from gnn_vuln_detection.naming_analysis.patterns import identify_naming_convention


def visualize_ast(ast_root: Node) -> None:
    """Visualize the AST structure."""
    graph = convert_ast_to_graph(ast_root)
    print("--- C Graph ---")

    print(f"Graph: {graph}")
    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())
    # for node in graph.nodes(data=True):
    #     print(f"Node: {node[0]}, Data: {node[1]}")

    # Display the graph using networkx and matplotlib
    plt.figure(figsize=(16, 8))
    pos = nx.spring_layout(graph, k=2, iterations=100, seed=42)

    # Draw nodes with better styling
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="lightsteelblue",
        node_size=800,
        alpha=0.9,
        edgecolors="navy",
        linewidths=1.5,
    )

    # Draw edges with better styling
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="darkgray",
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        width=1.2,
    )

    # Draw labels showing node_type property
    labels = {
        node: (
            data.get("node_type", str(node))[:15] + "..."
            if len(data.get("node_type", str(node))) > 15
            else data.get("node_type", str(node))
        )
        for node, data in graph.nodes(data=True)
    }
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=9,
        font_weight="bold",
        font_color="black",
    )

    plt.tight_layout()
    plt.title("AST Graph Visualization")
    plt.show()
    plt.clf()


def show_ast_structure(ast_root: Node) -> None:
    """Print the structure of the AST."""

    def print_node(node: Node, level: int = 0) -> None:
        indent = "  " * level
        print(f"{indent}{node.type} ({node.start_byte}-{node.end_byte})")
        for child in node.children:
            print_node(child, level + 1)

    print("AST Structure:")
    print_node(ast_root)


def analyze_c_code(code: str, language: str = "c"):
    """Analyze C code and return its AST representation."""
    parser = ASTParser(language=language)
    return parser.parse_code_to_ast(code)


def check_naming_conventions(ast_root: Node):
    """Check if the AST nodes follow naming conventions."""

    def check_node(node: Node):
        # print(f"Node type: {node.type}, text: {node.text.decode('utf-8') if node.text else None}")
        if node.type in {"function_definition", "function_declaration"}:
            function_name = node.child_by_field_name("name")
            print(f"{function_name=}")
            if function_name:
                return identify_naming_convention(function_name.text.decode("utf-8"))
        elif node.type in {"variable_definition", "variable_declaration"}:
            variable_name = node.child_by_field_name("name")
            print(f"{variable_name=}")
            if variable_name:
                return identify_naming_convention(variable_name.text.decode("utf-8"))
        elif node.type == "declaration":
            print(
                f"Declaration node found: {node.text.decode('utf-8') if node.text else None}",
            )
            for child in node.children:
                if child.type == "identifier":
                    print(f"Identifier found: {child.text.decode('utf-8')}")
                    return identify_naming_convention(child.text.decode("utf-8"))
        return None

    naming_conventions = []

    for child in ast_root.children:
        naming_conventions.append(check_node(child))

    return set(naming_conventions)


def convert_ast_to_graph(ast_root: Node):
    """Convert the AST to a graph representation."""
    graph_builder = GraphBuilder()
    return graph_builder.build_graph(ast_root)
    # stats = graph_builder.get_graph_statistics(graph)
    # print(stats)


def convert_ast_to_dataset(ast_root: Node) -> Data:
    """Convert the AST to a PyTorch Geometric Dataset."""
    converter = ASTToGraphConverter()
    return converter.ast_to_pyg_data(ast_root, label=0, include_edge_features=True)


def main() -> None:
    # Args
    visualize = False

    # Load sample C code from a data/raw/example_code directory
    # sample_code_dir = Path(__file__).parent / "data" / "raw" / "example_code"
    sample_code_dir = Path(__file__).parent.parent / "data" / "raw" / "example_code"
    c_files = {}

    if not sample_code_dir.exists():
        print(f"Sample code directory {sample_code_dir} does not exist.")
        return

    for file_path in sample_code_dir.glob("**/*.c"):
        with file_path.open("r", encoding="utf-8") as f:
            c_code = f.read()

        analyzed_code = analyze_c_code(c_code, language="c")
        conventions = check_naming_conventions(analyzed_code)
        pyg_data = convert_ast_to_dataset(analyzed_code)
        c_files[file_path.name] = {
            "code": c_code,
            "ast": analyzed_code,
            "naming_conventions": conventions,
            "pyg_data": pyg_data,
        }

        print(f"Analyzed AST for {file_path.name}:")
        print("Results")
        print("-" * 60)
        print(f"Naming conventions followed: {conventions}")
        print(f"PyTorch Geometric Data: {pyg_data}")
        # show_ast_structure(analyzed_code)
        if visualize:
            visualize_ast(analyzed_code)


if __name__ == "__main__":
    main()
