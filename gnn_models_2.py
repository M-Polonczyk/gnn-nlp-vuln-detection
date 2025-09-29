import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GCNConv, HeteroConv, SAGEConv, global_mean_pool

# Do obsługi komponentu Transformera (wymaga instalacji: pip install transformers)
# from transformers import AutoModel


# --- KROK 1: Definicja Enkodera Heterogenicznej Sieci GNN ---
# Ten moduł stanowi rdzeń GNN, przetwarzając graf i generując embeddingi dla każdego węzła.
class HeteroGNNEncoder(nn.Module):
    def __init__(
        self,
        node_types,
        edge_types,
        hidden_dim=256,
        out_dim=128,
        num_layers=3,
        heads=8,
        dropout=0.5,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()

        # Tworzenie warstw HeteroConv zgodnie ze specyfikacją
        for i in range(num_layers):
            in_channels = (
                hidden_dim if i > 0 else -1
            )  # -1 dla automatycznego dopasowania wymiaru wejściowego

            # Definicja różnych typów konwolucji dla różnych typów krawędzi
            conv_dict = {
                # Używamy GraphSAGE dla efektywnej agregacji z sąsiedztwa
                edge_type: SAGEConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    add_self_loops=False,
                )
                for edge_type in edge_types
            }
            # Dla wybranych relacji można użyć GAT z mechanizmem uwagi
            # Przykład: ('ast', 'child_of', 'ast') - relacje składniowe
            if ("ast", "child_of", "ast") in edge_types:
                conv_dict[("ast", "child_of", "ast")] = GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                )

            # GCN może być użyty do podstawowej propagacji informacji
            if ("declaration", "references", "token") in edge_types:
                conv_dict[("declaration", "references", "token")] = GCNConv(
                    in_channels, hidden_dim,
                )

            conv = HeteroConv(conv_dict, aggr="sum")
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)
        # Warstwa wyjściowa do projekcji embeddingów na ostateczny wymiar
        self.out_lin = nn.Linear(
            hidden_dim * heads if "GATConv" in str(self.convs) else hidden_dim, out_dim,
        )

    def forward(self, x_dict, edge_index_dict):
        # Propagacja przez warstwy GNN
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # Funkcja aktywacji i normalizacja dla każdego typu węzła
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        # Opcjonalnie można dodać dropout po warstwach konwolucyjnych
        return {key: self.dropout(x) for key, x in x_dict.items()}



# --- KROK 2: Definicja Głowic dla Multi-Task Learning ---
# Ten moduł zawiera oddzielne warstwy klasyfikacyjne/regresyjne dla każdego zadania.
class MultiTaskHeads(nn.Module):
    def __init__(self, input_dim=128) -> None:
        super().__init__()
        # Zadania główne
        self.cve_detection = nn.Linear(
            input_dim, 1,
        )  # Klasyfikacja binarna (podatny/niepodatny)
        self.cwe_classification = nn.Linear(
            input_dim, 151,
        )  # Klasyfikacja wieloklasowa (>150 klas)
        self.logic_errors = nn.Linear(input_dim, 1)  # Klasyfikacja binarna

        # Zadania pomocnicze
        self.code_duplication = nn.Linear(
            input_dim, 1,
        )  # Similarity (np. jako regresja)
        self.complexity_analysis = nn.Linear(input_dim, 1)  # Regresja
        self.naming_issues = nn.Linear(input_dim, 1)  # Klasyfikacja binarna

    def forward(self, graph_embedding):
        return {
            "cve_detection": self.cve_detection(graph_embedding),
            "cwe_classification": self.cwe_classification(graph_embedding),
            "logic_errors": self.logic_errors(graph_embedding),
            "code_duplication": self.code_duplication(graph_embedding),
            "complexity_analysis": self.complexity_analysis(graph_embedding),
            "naming_issues": self.naming_issues(graph_embedding),
        }


# --- KROK 3: Połączenie komponentów w główny model ---
# Ta klasa integruje GNN, Transformer (opcjonalnie) i głowice zadaniowe.
class SourceCodeAnalysisModel(nn.Module):
    def __init__(
        self,
        node_types,
        edge_types,
        hidden_dim=256,
        gnn_out_dim=128,
        num_layers=3,
        heads=8,
        dropout=0.5,
        use_transformer=False,
    ) -> None:
        super().__init__()

        self.use_transformer = use_transformer

        # Komponent GNN
        self.gnn_encoder = HeteroGNNEncoder(
            node_types, edge_types, hidden_dim, gnn_out_dim, num_layers, heads, dropout,
        )

        # Komponent Transformer (opcjonalny)
        transformer_out_dim = 0
        if self.use_transformer:
            # Przykład użycia CodeBERT. Model ten musi być załadowany.
            # self.transformer = AutoModel.from_pretrained("microsoft/codebert-base")
            # transformer_out_dim = self.transformer.config.hidden_size # Zazwyczaj 768
            # Dla celów demonstracyjnych, symulujemy jego wyjście
            transformer_out_dim = 768
            self.dimension_aligner = nn.Linear(
                transformer_out_dim, gnn_out_dim,
            )  # Warstwa do dopasowania wymiarów

        # Głowice do zadań Multi-Task Learning
        # Wymiar wejściowy głowic zależy od tego, czy używamy fuzji z Transformerem
        fusion_dim = (
            gnn_out_dim * len(node_types) + gnn_out_dim
            if self.use_transformer
            else gnn_out_dim * len(node_types)
        )
        self.task_heads = MultiTaskHeads(input_dim=fusion_dim)

    def forward(self, hetero_data, transformer_input=None):
        # 1. Przetwarzanie przez GNN
        node_embeddings = self.gnn_encoder(
            hetero_data.x_dict, hetero_data.edge_index_dict,
        )

        # 2. Agregacja embeddingów węzłów do jednego wektora reprezentującego cały graf
        # Prosta strategia: uśrednienie embeddingów każdego typu węzła i konkatenacja
        aggregated_embeddings = []
        for node_type in hetero_data.node_types:
            # Używamy batch'a, aby poprawnie agregować dla każdego grafu w paczce
            batch = hetero_data[node_type].batch
            aggregated_embeddings.append(
                global_mean_pool(node_embeddings[node_type], batch),
            )

        gnn_graph_embedding = torch.cat(aggregated_embeddings, dim=1)

        # 3. Fuzja z komponentem Transformer (jeśli aktywny)
        if self.use_transformer:
            # W rzeczywistym scenariuszu:
            # transformer_output = self.transformer(**transformer_input).pooler_output
            # Symulacja wyjścia dla demonstracji
            batch_size = gnn_graph_embedding.size(0)
            transformer_output = torch.randn(
                batch_size, 768,
            )  # Symulowane wyjście [CLS] tokenu

            aligned_transformer_embedding = F.relu(
                self.dimension_aligner(transformer_output),
            )

            # Strategia Fuzji: Konkatenacja (Early/Late Fusion)
            final_graph_embedding = torch.cat(
                [gnn_graph_embedding, aligned_transformer_embedding], dim=1,
            )
        else:
            final_graph_embedding = gnn_graph_embedding

        # 4. Przekazanie połączonego embeddingu do głowic zadaniowych
        return self.task_heads(final_graph_embedding)



# --- KROK 4: Przykład użycia i definicja strat ---
if __name__ == "__main__":
    # Definicja typów węzłów i krawędzi (zgodnie ze specyfikacją)
    NODE_TYPES = ["token", "ast", "cfg", "pdg", "declaration"]
    EDGE_TYPES = [
        ("ast", "child_of", "ast"),
        ("token", "flows_to", "token"),  # Data-flow
        ("cfg", "next_instruction", "cfg"),  # Control-flow
        ("declaration", "calls", "declaration"),  # Call-graph
        ("token", "references", "declaration"),  # Reference
    ]

    # Inicjalizacja modelu
    model = SourceCodeAnalysisModel(
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        hidden_dim=256,
        gnn_out_dim=128,
        num_layers=3,
        heads=8,
        dropout=0.4,
        use_transformer=True,  # Włączamy architekturę hybrydową
    )

    print("Model został pomyślnie utworzony:")
    print(model)

    # Tworzenie przykładowego, małego grafu (dla jednego fragmentu kodu)
    # W praktyce te dane byłyby generowane przez narzędzie do analizy kodu
    data = HeteroData()

    # Przykładowe węzły (z losowymi cechami)
    data["token"].x = torch.randn(
        10, 128,
    )  # 10 węzłów tokenów, każdy z wektorem cech o dł. 128
    data["ast"].x = torch.randn(15, 128)  # 15 węzłów AST
    data["cfg"].x = torch.randn(8, 128)  # 8 węzłów CFG
    data["pdg"].x = torch.randn(5, 128)
    data["declaration"].x = torch.randn(3, 128)

    # Przykładowe krawędzie
    data["ast", "child_of", "ast"].edge_index = torch.randint(
        0, 15, (2, 20),
    )  # 20 krawędzi AST
    data["token", "flows_to", "token"].edge_index = torch.randint(0, 10, (2, 12))
    data["cfg", "next_instruction", "cfg"].edge_index = torch.randint(0, 8, (2, 7))
    data["declaration", "calls", "declaration"].edge_index = torch.randint(0, 3, (2, 2))
    data["token", "references", "declaration"].edge_index = torch.stack(
        [torch.randint(0, 10, (5,)), torch.randint(0, 3, (5,))],
    )

    print("\nPrzykładowy heterogeniczny graf:")
    print(data)

    # Symulacja przetwarzania wsadu (batch) dwóch takich samych grafów
    from torch_geometric.loader import DataLoader

    batch_data = next(iter(DataLoader([data, data], batch_size=2)))

    # Przetworzenie danych przez model
    outputs = model(
        batch_data, transformer_input="dummy",
    )  # Przekazujemy dane do modelu

    print("\nWyniki z modelu (dla batch_size=2):")
    for task_name, prediction in outputs.items():
        print(f"- Wynik zadania '{task_name}': {prediction.shape}")

    # Definicja wag zadań i funkcji strat (zgodnie ze specyfikacją)
    task_weights = {
        "cve_detection": 0.3,
        "cwe_classification": 0.25,
        "logic_errors": 0.2,
        "code_duplication": 0.1,
        "complexity_analysis": 0.1,
        "naming_issues": 0.05,
    }

    # Kryteria strat dla poszczególnych zadań
    loss_fns = {
        "cve_detection": nn.BCEWithLogitsLoss(),
        "cwe_classification": nn.CrossEntropyLoss(),
        "logic_errors": nn.BCEWithLogitsLoss(),
        "code_duplication": nn.MSELoss(),
        "complexity_analysis": nn.MSELoss(),
        "naming_issues": nn.BCEWithLogitsLoss(),
    }

    # Przykładowe etykiety
    labels = {
        "cve_detection": torch.rand(2, 1),
        "cwe_classification": torch.randint(0, 151, (2,)),
        "logic_errors": torch.rand(2, 1),
        "code_duplication": torch.rand(2, 1),
        "complexity_analysis": torch.rand(2, 1),
        "naming_issues": torch.rand(2, 1),
    }

    # Obliczenie całkowitej, ważonej straty
    total_loss = 0
    for task_name, pred in outputs.items():
        loss = loss_fns[task_name](pred, labels[task_name])
        weighted_loss = task_weights[task_name] * loss
        total_loss += weighted_loss

    print(f"\nObliczona przykładowa, ważona strata (loss): {total_loss.item()}")
