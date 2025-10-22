import re

import inflect
import nltk

from .patterns import identify_naming_conventions

nltk.download("words", quiet=True)
english_words = set(nltk.corpus.words.words())
inflect_engine = inflect.engine()


def is_meaningful(name: str):
    """Check if an identifier name is meaningful - contains English words.

    Args:
        name (str): name of the identifier to check.

    Returns:
        bool: True if the name is meaningful, False otherwise.
    """
    parts = re.split(r"_|(?=[A-Z])", name)
    parts = [p.lower() for p in parts if len(p) > 1]
    return any(
        part in english_words or inflect_engine.singular_noun(part) or part.isalpha()
        for part in parts
    )


def extract_identifiers(tree, source_code, language):
    cursor = tree.walk()
    to_visit = [cursor.node]
    identifiers = []
    id_nodes = {
        "c": {
            "identifier",
            "function_definition",
            "variable_declarator",
            "method_declaration",
        },
        "cpp": {
            "identifier",
            "function_definition",
            "variable_declarator",
            "method_declaration",
            "class_definition",
        },
        "java": {
            "identifier",
            "function_definition",
            "variable_declarator",
            "method_declaration",
            "class_definition",
        },
        "python": {
            "identifier",
            "function_definition",
            "variable_declarator",
            "method_declaration",
            "class_definition",
        },
    }
    valid_types = id_nodes.get(language.lower(), {"identifier"})
    while to_visit:
        node = to_visit.pop()
        if node.type in valid_types:
            ident = {
                "type": node.type,
                "start": node.start_byte,
                "end": node.end_byte,
                "code": source_code[node.start_byte : node.end_byte].decode(),
            }
            identifiers.append(ident)
        to_visit.extend(node.children)
    return identifiers


def extract_identifiers_from_node(node, source_code, language="c"):
    """Recursively extract identifiers from a given AST node."""
    identifiers = []
    id_nodes = {
        "c": {
            "identifier",
            "function_definition",
            "variable_declarator",
            "method_declaration",
        },
        "cpp": {
            "identifier",
            "function_definition",
            "variable_declarator",
            "method_declaration",
            "class_definition",
        },
        "java": {
            "identifier",
            "function_definition",
            "variable_declarator",
            "method_declaration",
            "class_definition",
        },
        "python": {
            "identifier",
            "function_definition",
            "variable_declarator",
            "method_declaration",
            "class_definition",
        },
    }
    valid_types = id_nodes.get(language.lower(), {"identifier"})
    if node.type in valid_types:
        if node.type == "identifier":
            convention = identify_naming_conventions(node.text.decode("utf-8"))
            print(
                f"Identifier '{node.text.decode('utf-8')}' follows convention: {convention}"
            )
        identifiers.append(
            {
                "type": node.type,
                "start": node.start_byte,
                "end": node.end_byte,
                "code": source_code[node.start_byte : node.end_byte].decode(),
            }
        )
    for child in node.children:
        identifiers.extend(extract_identifiers_from_node(child, source_code))
    return identifiers
