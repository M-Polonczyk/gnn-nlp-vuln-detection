from gnn_vuln_detection.nlp_integration.api_patterns import APIPatternLearner
from gnn_vuln_detection.nlp_integration.code_complexity_analyzer import (
    CodeComplexityAnalyzer,
)
from gnn_vuln_detection.nlp_integration.code_duplication import (
    SemanticCloneDetector,
)
from gnn_vuln_detection.nlp_integration.text_encoders import CodeEmbedder
from gnn_vuln_detection.nlp_integration.variable_analyzer import (
    VariableLifecycleAnalyzer,
)


class NLPSASTPipeline:
    def __init__(self) -> None:
        self.embedder = CodeEmbedder()
        self.api_learner = APIPatternLearner(self.embedder)
        self.clone_detector = SemanticCloneDetector(self.embedder)
        self.lifecycle_analyzer = VariableLifecycleAnalyzer()
        self.complexity_analyzer = CodeComplexityAnalyzer(self.embedder)

        # Symulacja bazy wiedzy
        self.clone_detector.load_known_vulnerabilities(
            [
                "void vul(char* s) { char buf[10]; strcpy(buf, s); }",  # Buffer overflow sample
                "void vul2() { char* p = malloc(10); free(p); *p = 'a'; }",  # UAF sample
            ]
        )

    def process_function(self, function_code, cfg_nodes=None, cfg_edges=None) -> dict:
        """
        Główna metoda przetwarzająca pojedynczą funkcję.
        Zwraca słownik cech gotowy do dołączenia do węzła w grafie GNN.
        """

        features = {}

        # 1. Base Embedding
        features["bert_embedding"] = self.embedder.get_embedding(function_code)

        # 2. API Analysis
        api_risks = self.api_learner.extract_api_context(function_code)
        agg_risk, context_emb = self.api_learner.get_aggregated_risk(api_risks)
        features["api_risk_score"] = agg_risk
        features["api_context_emb"] = context_emb

        # 3. Clone Detection
        sim_score, is_clone = self.clone_detector.detect_clones(function_code)
        features["vulnerability_similarity"] = sim_score

        # 4. Complexity
        comp_metrics = self.complexity_analyzer.get_complexity_features(function_code)
        features.update(comp_metrics)

        # 5. Lifecycle (Requires Graph)
        if cfg_nodes and cfg_edges:
            G = self.lifecycle_analyzer.build_cfg_from_joern(cfg_nodes, cfg_edges)
            # Przykładowa analiza dla zmiennej 'ptr' - w praktyce pętla po wszystkich zmiennych
            uaf_detected, uaf_score = self.lifecycle_analyzer.detect_use_after_free(
                G, "ptr"
            )
            features["uaf_risk"] = uaf_score
        else:
            features["uaf_risk"] = 0.0

        return features


def run(sample_code, mock_nodes, mock_edges) -> dict:
    pipeline = NLPSASTPipeline()
    return pipeline.process_function(sample_code, mock_nodes, mock_edges)


if __name__ == "__main__":
    # Przykładowy kod C++ do analizy
    sample_code = """
    void process_data(char* input) {
        char* buffer = (char*)malloc(100);
        if (input != NULL) {
            strcpy(buffer, input); // Potential overflow if input > 100
        }
        free(buffer);
        // ... complex logic ...
        printf("%s", buffer); // Use After Free!
    }
    """

    # Symulacja danych z parsera (np. Joern)
    mock_nodes = [
        {"id": 1, "code": "char* buffer = malloc(100);"},
        {"id": 2, "code": "if (input != NULL)"},
        {"id": 3, "code": "strcpy(buffer, input);"},
        {"id": 4, "code": "free(buffer);"},
        {"id": 5, "code": 'printf("%s", buffer);'},
    ]
    mock_edges = [
        (1, 2, "Flow"),
        (2, 3, "True"),
        (2, 4, "False"),
        (3, 4, "Flow"),
        (4, 5, "Flow"),
    ]

    result = run(sample_code, mock_nodes, mock_edges)

    print("\n--- Analiza Zakończona ---")
    print(f"API Risk Score: {result['api_risk_score']:.4f}")
    print(f"Clone Similarity: {result['vulnerability_similarity']:.4f}")
    print(f"Cyclomatic Complexity: {result['cyclomatic']}")
    print(f"Semantic Variance: {result['semantic_variance']:.4f}")
    print(f"Use-After-Free Risk: {result['uaf_risk']:.4f}")
    print(
        f"Final Feature Vector Size: {result['bert_embedding'].shape[0] + result['api_context_emb'].shape[0] + 5}"
    )
