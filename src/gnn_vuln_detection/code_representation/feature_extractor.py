"""Extract features from nodes/edges"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from tree_sitter import Node


@dataclass
class GraphFeatures:
    """Dataclass for extracted graph features."""

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray | None = None
    node_types: list[str] | None = None
    edge_types: list[str] | None = None
    global_features: np.ndarray | None = None


class NodeFeatureExtractor:
    """Extract features from graph nodes (AST nodes)"""

    def __init__(self) -> None:
        self.vocabulary = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.node_type_to_id = {}
        self.id_to_node_type = {}

    def build_vocabulary(self, graphs: list[nx.DiGraph]) -> None:
        """Build vocabulary from a list of graphs"""
        all_tokens = []
        all_node_types = []

        for graph in graphs:
            for _node_id, data in graph.nodes(data=True):
                # Extract tokens from text
                text = data.get("text", "")
                tokens = self._tokenize_text(text)
                all_tokens.extend(tokens)

                # Extract node types
                node_type = data.get("node_type", "unknown")
                all_node_types.append(node_type)

        # Build token vocabulary
        token_counts = Counter(all_tokens)
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1}
        for i, (token, _) in enumerate(token_counts.most_common(), 2):
            self.token_to_id[token] = i
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # Build node type vocabulary
        node_type_counts = Counter(all_node_types)
        self.node_type_to_id = {}
        for i, (node_type, _) in enumerate(node_type_counts.most_common()):
            self.node_type_to_id[node_type] = i
        self.id_to_node_type = {v: k for k, v in self.node_type_to_id.items()}

    def extract_node_features(self, node_data: dict[str, Any]) -> dict[str, Any]:
        """Extract comprehensive features from a node"""
        features = {}

        # Basic features
        features.update(self._extract_basic_features(node_data))

        # Structural features
        features.update(self._extract_structural_features(node_data))

        # Text features
        features.update(self._extract_text_features(node_data))

        # Syntactic features
        features.update(self._extract_syntactic_features(node_data))

        return features

    def _extract_basic_features(self, node_data: dict[str, Any]) -> dict[str, Any]:
        """Extract basic node features"""
        return {
            "node_type_id": self.node_type_to_id.get(
                node_data.get("node_type", "unknown"),
                0,
            ),
            "is_named": int(node_data.get("is_named", False)),
            "has_error": int(node_data.get("has_error", False)),
            "is_leaf": int(node_data.get("is_leaf", False)),
            "child_count": node_data.get("child_count", 0),
            "text_length": node_data.get("text_length", 0),
        }

    def _extract_structural_features(self, node_data: dict[str, Any]) -> dict[str, Any]:
        """Extract structural features"""
        start_point = node_data.get("start_point", (0, 0))
        end_point = node_data.get("end_point", (0, 0))

        return {
            "start_row": start_point[0] if isinstance(start_point, tuple) else 0,
            "start_col": start_point[1] if isinstance(start_point, tuple) else 0,
            "end_row": end_point[0] if isinstance(end_point, tuple) else 0,
            "end_col": end_point[1] if isinstance(end_point, tuple) else 0,
            "span_rows": (
                (end_point[0] - start_point[0])
                if isinstance(start_point, tuple) and isinstance(end_point, tuple)
                else 0
            ),
            "span_cols": (
                (end_point[1] - start_point[1])
                if isinstance(start_point, tuple) and isinstance(end_point, tuple)
                else 0
            ),
            "start_byte": node_data.get("start_byte", 0),
            "end_byte": node_data.get("end_byte", 0),
            "byte_length": node_data.get("end_byte", 0)
            - node_data.get("start_byte", 0),
        }

    def _extract_text_features(self, node_data: dict[str, Any]) -> dict[str, Any]:
        """Extract text-based features"""
        text = node_data.get("text", "")

        if not text:
            return {
                "token_ids": [],
                "num_tokens": 0,
                "has_keywords": 0,
                "has_operators": 0,
                "has_numbers": 0,
                "has_strings": 0,
            }

        tokens = self._tokenize_text(text)
        token_ids = [self.token_to_id.get(token, 1) for token in tokens]  # 1 is <UNK>

        return {
            "token_ids": token_ids,
            "num_tokens": len(tokens),
            "has_keywords": int(self._has_keywords(text)),
            "has_operators": int(self._has_operators(text)),
            "has_numbers": int(self._has_numbers(text)),
            "has_strings": int(self._has_strings(text)),
        }

    def _extract_syntactic_features(self, node_data: dict[str, Any]) -> dict[str, Any]:
        """Extract syntactic features based on node type"""
        node_type = node_data.get("node_type", "unknown")

        return {
            "is_statement": int(self._is_statement(node_type)),
            "is_expression": int(self._is_expression(node_type)),
            "is_declaration": int(self._is_declaration(node_type)),
            "is_control_flow": int(self._is_control_flow(node_type)),
            "is_function_related": int(self._is_function_related(node_type)),
            "is_variable_related": int(self._is_variable_related(node_type)),
        }

    def _tokenize_text(self, text: str) -> list[str]:
        """Tokenize text into meaningful tokens"""
        if not text:
            return []

        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return [token.lower() for token in tokens if token.strip()]

    def _has_keywords(self, text: str) -> bool:
        """Check if text contains programming keywords"""
        keywords = {
            "if",
            "else",
            "for",
            "while",
            "do",
            "switch",
            "case",
            "default",
            "return",
            "break",
            "continue",
            "goto",
            "int",
            "char",
            "float",
            "double",
            "void",
            "struct",
            "union",
            "enum",
            "typedef",
            "static",
            "extern",
            "const",
            "volatile",
            "auto",
            "register",
            "sizeof",
            "malloc",
            "free",
            "printf",
            "scanf",
            "include",
            "define",
        }
        tokens = set(self._tokenize_text(text.lower()))
        return bool(tokens.intersection(keywords))

    def _has_operators(self, text: str) -> bool:
        """Check if text contains operators"""
        operators = {
            "+",
            "-",
            "*",
            "/",
            "%",
            "=",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "&&",
            "||",
            "!",
            "&",
            "|",
            "^",
            "~",
            "<<",
            ">>",
            "++",
            "--",
            "+=",
            "-=",
            "*=",
            "/=",
        }
        return any(op in text for op in operators)

    def _has_numbers(self, text: str) -> bool:
        """Check if text contains numbers"""
        return bool(re.search(r"\d", text))

    def _has_strings(self, text: str) -> bool:
        """Check if text contains string literals"""
        return '"' in text or "'" in text

    def _is_statement(self, node_type: str) -> bool:
        """Check if node type is a statement"""
        statement_types = {
            "expression_statement",
            "if_statement",
            "while_statement",
            "for_statement",
            "do_statement",
            "switch_statement",
            "return_statement",
            "break_statement",
            "continue_statement",
            "goto_statement",
            "labeled_statement",
            "compound_statement",
        }
        return node_type in statement_types

    def _is_expression(self, node_type: str) -> bool:
        """Check if node type is an expression"""
        expression_types = {
            "binary_expression",
            "unary_expression",
            "assignment_expression",
            "call_expression",
            "conditional_expression",
            "parenthesized_expression",
            "subscript_expression",
            "field_expression",
            "cast_expression",
            "sizeof_expression",
            "identifier",
            "number_literal",
            "string_literal",
        }
        return node_type in expression_types

    def _is_declaration(self, node_type: str) -> bool:
        """Check if node type is a declaration"""
        declaration_types = {
            "declaration",
            "function_definition",
            "parameter_declaration",
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
            "typedef_declaration",
        }
        return node_type in declaration_types

    def _is_control_flow(self, node_type: str) -> bool:
        """Check if node type affects control flow"""
        control_flow_types = {
            "if_statement",
            "while_statement",
            "for_statement",
            "do_statement",
            "switch_statement",
            "return_statement",
            "break_statement",
            "continue_statement",
            "goto_statement",
        }
        return node_type in control_flow_types

    def _is_function_related(self, node_type: str) -> bool:
        """Check if node type is function related"""
        function_types = {
            "function_definition",
            "function_declarator",
            "call_expression",
            "parameter_list",
            "parameter_declaration",
        }
        return node_type in function_types

    def _is_variable_related(self, node_type: str) -> bool:
        """Check if node type is variable related"""
        variable_types = {
            "identifier",
            "declaration",
            "assignment_expression",
            "init_declarator",
            "declarator",
        }
        return node_type in variable_types


class EdgeFeatureExtractor:
    """Extract features from graph edges"""

    def __init__(self) -> None:
        self.edge_type_to_id = {}
        self.id_to_edge_type = {}

    def build_edge_vocabulary(self, graphs: list[nx.DiGraph]) -> None:
        """Build edge type vocabulary from graphs"""
        all_edge_types = []

        for graph in graphs:
            for _u, _v, data in graph.edges(data=True):
                edge_type = data.get("edge_type", "unknown")
                all_edge_types.append(edge_type)

        edge_type_counts = Counter(all_edge_types)
        self.edge_type_to_id = {}
        for i, (edge_type, _) in enumerate(edge_type_counts.most_common()):
            self.edge_type_to_id[edge_type] = i
        self.id_to_edge_type = {v: k for k, v in self.edge_type_to_id.items()}

    def extract_edge_features(
        self,
        edge_data: dict[str, Any],
        source_node: dict[str, Any],
        target_node: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract features from an edge"""
        features = {}

        # Edge type
        edge_type = edge_data.get("edge_type", "unknown")
        features["edge_type_id"] = self.edge_type_to_id.get(edge_type, 0)

        # Distance features
        features.update(self._extract_distance_features(source_node, target_node))

        # Relationship features
        features.update(
            self._extract_relationship_features(edge_data, source_node, target_node),
        )

        return features

    def _extract_distance_features(
        self,
        source_node: dict[str, Any],
        target_node: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract distance-based features"""
        source_start = source_node.get("start_point", (0, 0))
        target_start = target_node.get("start_point", (0, 0))

        if isinstance(source_start, tuple) and isinstance(target_start, tuple):
            row_distance = abs(target_start[0] - source_start[0])
            col_distance = abs(target_start[1] - source_start[1])
        else:
            row_distance = 0
            col_distance = 0

        source_byte = source_node.get("start_byte", 0)
        target_byte = target_node.get("start_byte", 0)
        byte_distance = abs(target_byte - source_byte)

        return {
            "row_distance": row_distance,
            "col_distance": col_distance,
            "byte_distance": byte_distance,
            "is_forward": int(target_byte > source_byte),
        }

    def _extract_relationship_features(
        self,
        edge_data: dict[str, Any],
        source_node: dict[str, Any],
        target_node: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract relationship features"""
        edge_type = edge_data.get("edge_type", "unknown")

        return {
            "is_ast_edge": int(edge_type == "ast"),
            "is_control_flow": int(edge_type == "control_flow"),
            "is_data_flow": int(edge_type == "data_flow"),
            "is_call_edge": int(edge_type == "call"),
            "is_next_token": int(edge_type == "next_token"),
            "same_node_type": int(
                source_node.get("node_type") == target_node.get("node_type"),
            ),
        }


class GraphFeatureExtractor:
    """Extract global graph-level features"""

    def __init__(self) -> None:
        pass

    def extract_graph_features(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Extract graph-level features"""
        features = {}

        # Basic graph statistics
        features.update(self._extract_basic_graph_features(graph))

        # Structural features
        features.update(self._extract_structural_graph_features(graph))

        # Node type distribution
        features.update(self._extract_node_type_distribution(graph))

        # Edge type distribution
        features.update(self._extract_edge_type_distribution(graph))

        return features

    def _extract_basic_graph_features(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Extract basic graph features"""
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_empty": int(graph.number_of_nodes() == 0),
        }

    def _extract_structural_graph_features(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Extract structural graph features"""
        features = {}

        if graph.number_of_nodes() > 0:
            # Connectivity
            features["is_weakly_connected"] = int(nx.is_weakly_connected(graph))
            features["num_weakly_connected_components"] = (
                nx.number_weakly_connected_components(graph)
            )

            # Tree-like properties
            features["is_tree"] = int(nx.is_tree(graph.to_undirected()))
            features["is_forest"] = int(nx.is_forest(graph.to_undirected()))

            # Depth (for tree-like structures)
            if nx.is_weakly_connected(graph):
                try:
                    # Find a root-like node (one with no predecessors)
                    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
                    if roots:
                        root = roots[0]
                        shortest_paths = nx.single_source_shortest_path_length(
                            graph,
                            root,
                        )
                        features["max_depth"] = (
                            max(shortest_paths.values()) if shortest_paths else 0
                        )
                        features["avg_depth"] = (
                            np.mean(list(shortest_paths.values()))
                            if shortest_paths
                            else 0
                        )
                    else:
                        features["max_depth"] = 0
                        features["avg_depth"] = 0
                except:  # noqa
                    features["max_depth"] = 0
                    features["avg_depth"] = 0
            else:
                features["max_depth"] = 0
                features["avg_depth"] = 0
        else:
            features.update(
                {
                    "is_weakly_connected": 0,
                    "num_weakly_connected_components": 0,
                    "is_tree": 0,
                    "is_forest": 0,
                    "max_depth": 0,
                    "avg_depth": 0,
                },
            )

        return features

    def _extract_node_type_distribution(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Extract node type distribution features"""
        node_types = []
        for _, data in graph.nodes(data=True):
            node_types.append(data.get("node_type", "unknown"))

        type_counts = Counter(node_types)
        total_nodes = len(node_types)

        features = {}
        # Top node types (up to 10 most common)
        for i, (_node_type, count) in enumerate(type_counts.most_common(10)):
            features[f"node_type_{i}_ratio"] = (
                count / total_nodes if total_nodes > 0 else 0
            )

        features["num_unique_node_types"] = len(type_counts)

        return features

    def _extract_edge_type_distribution(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Extract edge type distribution features"""
        edge_types = []
        for _, _, data in graph.edges(data=True):
            edge_types.append(data.get("edge_type", "unknown"))

        type_counts = Counter(edge_types)
        total_edges = len(edge_types)

        features = {}
        # Top edge types (up to 5 most common)
        for i, (_edge_type, count) in enumerate(type_counts.most_common(5)):
            features[f"edge_type_{i}_ratio"] = (
                count / total_edges if total_edges > 0 else 0
            )

        features["num_unique_edge_types"] = len(type_counts)

        return features


class ASTFeatureExtractor:
    """Extract features from AST nodes for GNN models"""

    def __init__(self) -> None:
        self.node_type_vocab = {}
        self.max_features = 64  # Fixed feature dimension

        # Common C language node types with vulnerability relevance
        self.important_node_types = {
            "function_definition": 0,
            "function_declarator": 1,
            "call_expression": 2,
            "identifier": 3,
            "string_literal": 4,
            "number_literal": 5,
            "assignment_expression": 6,
            "binary_expression": 7,
            "if_statement": 8,
            "while_statement": 9,
            "for_statement": 10,
            "declaration": 11,
            "parameter_list": 12,
            "argument_list": 13,
            "compound_statement": 14,
            "return_statement": 15,
            "pointer_declarator": 16,
            "array_declarator": 17,
            "struct_specifier": 18,
            "union_specifier": 19,
            "cast_expression": 20,
        }

        # Vulnerability-related function patterns
        self.vuln_functions = {
            "strcpy",
            "strcat",
            "sprintf",
            "gets",
            "scanf",
            "system",
            "exec",
            "eval",
            "malloc",
            "free",
            "memcpy",
            "memmove",
        }

    def extract_features_from_ast(
        self,
        ast_root: Node,
    ) -> tuple[list[list[float]], dict[Node, int]]:
        """
        Extract features from AST and return node features and mapping.

        Returns:
            Tuple of (node_features, node_mapping) where:
            - node_features: List of feature vectors for each node
            - node_mapping: Dict mapping AST nodes to indices
        """
        node_features = []
        node_mapping = {}

        # Traverse AST and extract features for each node
        nodes = list(self._traverse_ast(ast_root))

        for i, node in enumerate(nodes):
            node_mapping[node] = i
            features = self._extract_node_features(node)
            node_features.append(features)

        return node_features, node_mapping

    def _traverse_ast(self, node: Node):
        """Traverse AST in depth-first order"""
        yield node
        for child in node.children:
            yield from self._traverse_ast(child)

    def _extract_node_features(self, node: Node) -> list[float]:
        """Extract features for a single AST node"""
        features = [0.0] * self.max_features

        # 1. Node type features (first 21 positions)
        node_type_id = self.important_node_types.get(node.type, 21)  # 21 for 'other'
        if node_type_id < 21:
            features[node_type_id] = 1.0

        # 2. Structural features (positions 22-35)
        features[22] = len(node.children)  # Number of children
        features[23] = 1.0 if node.is_named else 0.0  # Is named node
        features[24] = node.start_point[0] if node.start_point else 0  # Start line
        features[25] = node.start_point[1] if node.start_point else 0  # Start column
        features[26] = node.end_point[0] if node.end_point else 0  # End line
        features[27] = node.end_point[1] if node.end_point else 0  # End column

        # Calculate span
        if node.start_point and node.end_point:
            features[28] = node.end_point[0] - node.start_point[0]  # Line span
            features[29] = node.end_point[1] - node.start_point[1]  # Column span

        # 3. Text-based features (positions 30-45)
        if node.text:
            text = (
                node.text.decode("utf-8")
                if isinstance(node.text, bytes)
                else str(node.text)
            )
            features[30] = len(text)  # Text length
            features[31] = 1.0 if self._contains_vuln_function(text) else 0.0
            features[32] = 1.0 if self._contains_buffer_operations(text) else 0.0
            features[33] = 1.0 if self._contains_memory_operations(text) else 0.0
            features[34] = 1.0 if self._contains_input_operations(text) else 0.0
            features[35] = 1.0 if self._contains_format_strings(text) else 0.0
            features[36] = 1.0 if self._contains_pointers(text) else 0.0
            features[37] = 1.0 if self._contains_arrays(text) else 0.0
            features[38] = text.count("*")  # Pointer dereferences
            features[39] = text.count("[")  # Array accesses
            features[40] = text.count("(")  # Function calls

            # Simple pattern matching
            features[41] = 1.0 if "if" in text.lower() else 0.0
            features[42] = 1.0 if "for" in text.lower() else 0.0
            features[43] = 1.0 if "while" in text.lower() else 0.0
            features[44] = 1.0 if "return" in text.lower() else 0.0
            features[45] = (
                1.0
                if any(op in text for op in ["==", "!=", "<", ">", "<=", ">="])
                else 0.0
            )

        # 4. Context features (positions 46-55)
        parent = node.parent if hasattr(node, "parent") else None
        if parent:
            features[46] = 1.0 if parent.type == "function_definition" else 0.0
            features[47] = 1.0 if parent.type == "if_statement" else 0.0
            features[48] = 1.0 if parent.type == "while_statement" else 0.0
            features[49] = 1.0 if parent.type == "for_statement" else 0.0

        # 5. Depth and position features (positions 56-63)
        depth = self._calculate_depth(node)
        features[56] = min(depth / 10.0, 1.0)  # Normalized depth

        # Sibling information
        if parent:
            sibling_count = len(parent.children)
            features[57] = min(sibling_count / 10.0, 1.0)  # Normalized sibling count

            # Position among siblings
            try:
                position = list(parent.children).index(node)
                features[58] = position / max(
                    sibling_count - 1,
                    1,
                )  # Normalized position
            except (ValueError, ZeroDivisionError):
                features[58] = 0.0

        # Additional vulnerability indicators
        features[59] = (
            1.0 if node.type in ["call_expression", "function_definition"] else 0.0
        )
        features[60] = 1.0 if node.type in ["string_literal", "char_literal"] else 0.0
        features[61] = (
            1.0 if node.type in ["assignment_expression", "update_expression"] else 0.0
        )
        features[62] = 1.0 if "error" in node.type.lower() else 0.0
        features[63] = 1.0 if node.is_missing else 0.0

        return features

    def _calculate_depth(self, node: Node) -> int:
        """Calculate depth of node in AST"""
        depth = 0
        current = node
        while hasattr(current, "parent") and current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def _contains_vuln_function(self, text: str) -> bool:
        """Check if text contains vulnerable function calls"""
        text_lower = text.lower()
        return any(func in text_lower for func in self.vuln_functions)

    def _contains_buffer_operations(self, text: str) -> bool:
        """Check for buffer-related operations"""
        buffer_ops = ["strcpy", "strcat", "sprintf", "strncpy", "strncat"]
        text_lower = text.lower()
        return any(op in text_lower for op in buffer_ops)

    def _contains_memory_operations(self, text: str) -> bool:
        """Check for memory-related operations"""
        memory_ops = [
            "malloc",
            "free",
            "calloc",
            "realloc",
            "memcpy",
            "memmove",
            "memset",
        ]
        text_lower = text.lower()
        return any(op in text_lower for op in memory_ops)

    def _contains_input_operations(self, text: str) -> bool:
        """Check for input operations"""
        input_ops = ["gets", "scanf", "fgets", "getchar", "fread"]
        text_lower = text.lower()
        return any(op in text_lower for op in input_ops)

    def _contains_format_strings(self, text: str) -> bool:
        """Check for format string usage"""
        return "%" in text and any(
            spec in text for spec in ["%s", "%d", "%c", "%x", "%p"]
        )

    def _contains_pointers(self, text: str) -> bool:
        """Check for pointer usage"""
        return "*" in text or "->" in text

    def _contains_arrays(self, text: str) -> bool:
        """Check for array usage"""
        return "[" in text and "]" in text
