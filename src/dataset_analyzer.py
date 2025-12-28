#!/usr/bin/env python3
"""Example."""

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.code_representation.code_representation import CodeSample
from gnn_vuln_detection.data_processing.graph_converter import DataclassToGraphConverter
from gnn_vuln_detection.dataset.loaders import DiverseVulDatasetLoader
from gnn_vuln_detection.utils import config_loader


def analyze_codesamples_distribution(samples: list[CodeSample], cwe_mapping: dict):
    """
    Analizuje rozkład etykiet w liście obiektów CodeSample.
    cwe_mapping: słownik {indeks: "Nazwa CWE"}
    """

    # 1. Konwersja etykiet do macierzy numpy
    # Odfiltrowujemy próbki, które nie mają jeszcze przypisanych etykiet binarnych
    valid_labels = [s.cwe_ids_labeled for s in samples if s.cwe_ids_labeled is not None]

    if not valid_labels:
        print(
            "Błąd: Żaden z obiektów CodeSample nie posiada wypełnionego pola 'cwe_ids_labeled'."
        )
        return None

    y_matrix = np.array(valid_labels)
    num_samples = y_matrix.shape[0]
    num_classes = y_matrix.shape[1]

    # 2. Obliczanie statystyk dla klas (kolumny)
    class_counts = y_matrix.sum(axis=0)

    stats_data = []
    for i in range(num_classes):
        count = int(class_counts[i])
        stats_data.append(
            {
                "Index": i,
                "CWE": cwe_mapping.get(i, f"Idx_{i}"),
                "Count": count,
                "Percentage": f"{(count / num_samples * 100):.2f}%",
                "Pos_Weight": round((num_samples - count) / (count + 1e-6), 2),
            }
        )

    df_stats = pd.DataFrame(stats_data).sort_values(by="Count", ascending=False)

    # 3. Analiza gęstości (ile etykiet na jedną funkcję/próbkę)
    labels_per_sample = y_matrix.sum(axis=1)
    density_counts = Counter(labels_per_sample)

    # --- WYŚWIETLANIE RAPORTU ---
    print("\n" + "=" * 60)
    print(f"RAPORT ZBIORU DANYCH (Próbek: {num_samples})")
    print("=" * 60)
    print(df_stats.to_string(index=False))

    print("\n" + "-" * 30)
    print("ROZKŁAD LICZBY PODATNOŚCI NA PRÓBKĘ:")
    for n_labels in sorted(density_counts.keys()):
        count = density_counts[n_labels]
        print(
            f" {int(n_labels)} CWE: {count} próbek ({count / num_samples * 100:.1f}%)"
        )
    print("-" * 30)

    # 4. WYKRES Z WARTOŚCIAMI LICZBOWYMI
    plt.figure(figsize=(14, 7))
    bars = plt.bar(
        df_stats["CWE"],
        df_stats["Count"],
        color="royalblue",
        edgecolor="black",
        alpha=0.8,
    )

    # Dodawanie wartości liczbowych nad słupkami
    plt.bar_label(bars, padding=3, fontsize=10, fontweight="bold")

    plt.title(f"Częstotliwość występowania CWE (N={num_samples})", fontsize=14)
    plt.ylabel("Liczba wystąpień", fontsize=12)
    plt.xlabel("Identyfikator CWE", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    return df_stats


def analyze_dataset_labels(loader, cwe_mapping):
    """
    Analizuje rozkład etykiet multi-label w DataLoaderze.
    cwe_mapping: słownik {indeks: "CWE-ID"}
    """
    all_y = []
    for data in loader:
        all_y.append(data.y.cpu())

    # Łączymy wszystkie etykiety w jedną macierz [N, num_classes]
    y_matrix = torch.cat(all_y, dim=0).numpy()
    num_samples = y_matrix.shape[0]
    num_classes = y_matrix.shape[1]

    # 1. Obliczamy statystyki dla każdej klasy
    class_counts = y_matrix.sum(axis=0)
    class_percentages = (class_counts / num_samples) * 100

    stats = []
    for i in range(num_classes):
        stats.append(
            {
                "Index": i,
                "CWE": cwe_mapping.get(i, f"Class_{i}"),
                "Count": int(class_counts[i]),
                "Percentage (%)": round(class_percentages[i], 4),
                "Pos_Weight (Simple)": round(
                    (num_samples - class_counts[i]) / (class_counts[i] + 1e-6), 2
                ),
            }
        )

    df_stats = pd.DataFrame(stats).sort_values(by="Count", ascending=False)

    # 2. Analiza gęstości etykiet (ile etykiet na jedną próbkę kodu)
    labels_per_sample = y_matrix.sum(axis=1)
    unique, counts = np.unique(labels_per_sample, return_counts=True)
    label_density = dict(zip(unique.astype(int), counts, strict=False))

    # --- WYŚWIETLANIE WYNIKÓW ---
    print("\n" + "=" * 50)
    print(f"ANALIZA DATASETU (Suma próbek: {num_samples})")
    print("=" * 50)
    print(df_stats.to_string(index=False))

    print("\n" + "=" * 50)
    print("ROZKŁAD LICZBY ETYKIET NA PRÓBKĘ")
    print("=" * 50)
    for num_labels, count in label_density.items():
        print(
            f"{num_labels} etykiet(y): {count} próbek ({round(count / num_samples * 100, 2)}%)"
        )

    # 3. Wykres słupkowy
    plt.figure(figsize=(12, 6))
    plt.bar(df_stats["CWE"], df_stats["Count"], color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Liczba wystąpień poszczególnych CWE w zbiorze")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()
    plt.show()

    return df_stats, label_density


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_config = config_loader.load_config("dataset_paths.yaml")
    model_params = config_loader.load_config("model_params.yaml")["gcn_multiclass"]
    cwes = config_loader.load_config("model_params.yaml")["vulnerabilities"]

    num_classes = model_params["num_classes"]
    cwe_to_index = {val["cwe_id"]: val["index"] for val in cwes}
    index_to_cwe = {v: k for k, v in cwe_to_index.items()}

    train_loader = torch.load(
        "data/processed/test-diversevul-small-c.pt",
        weights_only=False,
    )
    # analyze_dataset_labels(train_loader, index_to_cwe)

    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )

    converter = DataclassToGraphConverter()
    ast_parser = converter.ast_parser
    samples = diversevul_loader.load_dataset(list(cwe_to_index.keys()))
    for sample in tqdm(samples, desc="Converting samples to nx graphs"):
        label_vec = [0] * num_classes
        if sample.cwe_ids:
            for cwe in sample.cwe_ids:
                if cwe in cwe_to_index:
                    label_vec[cwe_to_index[cwe]] = 1
        sample.cwe_ids_labeled = label_vec
    df_results = analyze_codesamples_distribution(samples, index_to_cwe)


if __name__ == "__main__":
    main()
