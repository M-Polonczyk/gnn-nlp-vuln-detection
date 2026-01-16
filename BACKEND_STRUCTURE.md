# Struktura i Logika Backendu - GNN-NLP Vulnerability Detection

Ten dokument zawiera podsumowanie architektury backendu projektu służącego do wykrywania podatności w kodzie przy użyciu Grafowych Sieci Neuronowych (GNN) oraz Przetwarzania Języka Naturalnego (NLP).

## 1. Przegląd Struktury Katalogów

Repozytorium zorganizowane jest w sposób modułowy, oddzielając logikę przetwarzania danych, definicje modeli i skrypty uruchomieniowe.

```
src/
├── gnn_vuln_detection/          # Główny pakiet biblioteki
│   ├── code_representation/     # Parsowanie kodu i budowa grafów (reprezentacja pośrednia)
│   ├── data_processing/         # Konwersja do formatów tensorowych (PyTorch Geometric)
│   ├── dataset/                 # Loadery dla zbiorów danych (DiverseVul, MegaVul)
│   ├── models/                  # Architektury sieci neuronowych (GNN)
│   ├── naming_analysis/         # Analiza konwencji nazewniczych zmiennych/funkcji
│   ├── training/                # Pętle treningowe i walidacyjne
│   └── utils/                   # Narzędzia konfiguracyjne i pomocnicze
├── process_dataset.py           # Skrypt orkiestrujący przetwarzanie datasetów
├── process_dataset_workers.py   # Wersja wielowątkowa/zrównoleglona przetwarzania
├── train.py                     # Skrypt treningowy (wewnątrz src)
└── main.py                      # Prosty punkt wejścia / demo analizy
```

## 2. Kluczowe Moduły

### Reprezentacja Kodu (`src/gnn_vuln_detection/code_representation/`)

Ten moduł odpowiada za przekształcenie surowego kodu źródłowego (głównie C/C++) w format zrozumiały dla algorytmów.

- **`ast_parser.py`**: Parsuje kod źródłowy do drzewa składniowego (AST).
- **`graph_builder.py`**: Buduje grafową reprezentację kodu na podstawie AST.
- **`feature_extractor.py`**: Ekstrahuje cechy z węzłów kodu (np. typy instrukcji).

### Modele GNN (`src/gnn_vuln_detection/models/`)

Definicje architektur sieci neuronowych.

- **`gnn/standard.py`**: Zawiera klasę `VulnerabilityGCN`, implementującą sieć splotową (GCNConv) z warstwami Batch Normalization i poolingiem (mean/max).
- **`gnn/gat.py`**: Implementacja Graph Attention Network (GAT).
- **`factory.py`**: Fabryka modeli ułatwiająca inicjalizację odpowiedniej architektury na podstawie konfiguracji.

### Przetwarzanie Danych (`src/process_dataset.py`, `src/gnn_vuln_detection/dataset/`)

- **Loadery**: Obsługa specyficznych formatów datasetów (DiverseVul).
- **Podział danych**: Implementacja stratyfikowanego podziału (iterative stratification) dla problemów multilabel (wiele typów CWE jednocześnie).
- **Pipelines**: Przekształcanie obiektów `CodeSample` w grafy PyTorch Geometric (`Data` objects).

## 3. Przepływ Danych (Data Pipeline)

Proces przygotowania danych wygląda następująco:

1. **Wczytanie**: Surowy kod jest wczytywany z plików datasetu (np. JSONL).
2. **Parsowanie**: Kod jest parsowany do AST.
3. **Ekstrakcja Cech**:
    - Cechy strukturalne z AST.
    - Cechy semantyczne (embeddingi) z CodeBERT.
4. **Budowa Grafu**: Tworzony jest graf, gdzie węzły to instrukcje/tokeny, a krawędzie to relacje w kodzie.
5. **Konwersja**: Graf jest konwertowany do obiektu `torch_geometric.data.Data`.
6. **Serializacja**: Przetworzone grafy są zapisywane (cache'owane) do plików `.pt` w katalogu `data/processed/`, aby przyspieszyć trening.

## 4. Technologie

- **PyTorch**: Główny framework DL.
- **PyTorch Geometric (PyG)**: Biblioteka do obsługi GNN.
- **Transformers (Hugging Face)**: Obsługa modelu CodeBERT.
- **Scikit-multilearn**: Do stratyfikowanego podziału zbiorów multilabel.
- **Hydra/OmegaConf / YAML**: Zarządzanie konfiguracją (pliki w folderze `config/`).
