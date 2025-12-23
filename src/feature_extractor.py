import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.code_representation.feature_extractor import CodeGraphProcessor

# from gnn_vuln_detection.code_representation.feature_extractor import ASTFeatureExtractor

if __name__ == "__main__":
    # basic_extractor = ASTFeatureExtractor(max_nodes=500, embedding_dim=128)

    # # Example source code
    # code_sample = """
    # def calculate_sum(a, b):
    #     result = a + b
    #     if result > 100:
    #         return result
    #     else:
    #         return 0
    # """

    # # Extract basic features
    # features = basic_extractor.extract_features(code_sample)
    # print(f"Basic feature array shape: {features.shape}")
    # Tworzymy mockowy graf
    G = nx.DiGraph()
    G.add_node(
        1,
        node_type="function_definition",
        text="void bug() { strcpy(a, b); }",
        start_byte=0,
    )
    G.add_node(2, node_type="call_expression", text="strcpy(a, b)", start_byte=15)
    G.add_edge(1, 2, edge_type="ast")

    processor = CodeGraphProcessor(node_dim=64)
    processor.fit([G])  # W realnym scenariuszu podaj listę grafów treningowych

    features = processor.process(G)

    print("Node Features Shape:", features.node_features.shape)
    print("Edge Index:\n", features.edge_index)
    print(
        "Vulnerability detected in features:", features.node_features[1, 27]
    )  # Pattern strcpy
