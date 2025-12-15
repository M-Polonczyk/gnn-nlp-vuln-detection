import networkx as nx


class VariableLifecycleAnalyzer:
    def __init__(self):
        pass

    def build_cfg_from_joern(self, nodes, edges):
        """
        Helper: Tworzy obiekt NetworkX DiGraph z danych (np. CSV z Joern).
        nodes: list of dicts {'id': int, 'code': str, 'type': str}
        edges: list of tuples (source_id, target_id, type)
        """
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n["id"], code=n.get("code", ""), type=n.get("type", "Statement"))
        for u, v, t in edges:
            G.add_edge(u, v, type=t)
        return G

    def detect_use_after_free(self, cfg_graph, var_name):
        """
        Wykrywa ścieżkę: free(x) -> ... -> use(x)
        Zwraca: (True/False, Confidence Score)
        """
        free_nodes = []
        use_nodes = []

        # 1. Znajdź węzły z free() i użyciem zmiennej
        for node_id, data in cfg_graph.nodes(data=True):
            code = data.get("code", "")
            # Prosta heurystyka regexowa (w produkcji używamy AST node type)
            if f"free({var_name})" in code:
                free_nodes.append(node_id)
            elif var_name in code and "free" not in code:  # Uproszczenie
                use_nodes.append(node_id)

        if not free_nodes or not use_nodes:
            return False, 0.0

        # 2. Sprawdź osiągalność (Reachability) w grafie CFG
        risk_score = 0.0
        detected = False

        for f_node in free_nodes:
            for u_node in use_nodes:
                # Jeśli istnieje ścieżka od free do use
                if nx.has_path(cfg_graph, f_node, u_node):
                    detected = True
                    # Im krótsza ścieżka, tym łatwiejszy exploit (wyższe ryzyko)
                    shortest_path = nx.shortest_path_length(cfg_graph, f_node, u_node)
                    risk_score = max(risk_score, 1.0 / (shortest_path + 1))

        return detected, risk_score

    def analyze_taint(
        self, cfg_graph, var_name, source_functions=["gets", "scanf", "recv"]
    ):
        """
        Sprawdza czy zmienna pochodzi od niebezpiecznego źródła (taint source).
        Analiza wsteczna (Data Flow).
        """
        # Znajdź definicję zmiennej (uproszczone: szukamy 'var_name = ...')
        def_nodes = [
            n
            for n, d in cfg_graph.nodes(data=True)
            if f"{var_name} =" in d.get("code", "")
        ]

        taint_score = 0.0
        for def_node in def_nodes:
            code = cfg_graph.nodes[def_node]["code"]
            # Czy definicja zawiera źródło taint?
            if any(src in code for src in source_functions):
                return 1.0  # High confidence taint

            # (Tutaj w pełnej wersji: rekursywne sprawdzanie poprzedników w DFG)

        return taint_score
