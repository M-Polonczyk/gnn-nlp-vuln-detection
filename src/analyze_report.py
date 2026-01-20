import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MetricsAnalyzer")

PLOTS_DIR = Path("comparison/plots")
REPORT_JSON_FILE = Path("comparison/output.json")

TOOLS = {
    "cwes_corgea": "Corgea",
    "cwes_aikido": "Aikido",
    "cwes_semgrep": "Semgrep",
    "cwes_gnn": "GNN Model",
}


def ensure_list(x) -> list:
    if isinstance(x, str):
        return [x] if x else []
    return x if x is not None else []


def calculate_binary_metrics(y_true, y_pred):
    """Calculates binary classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_comparison_bar(df_metrics, title, save_path):
    """Plots a bar chart comparing tools."""
    plt.figure(figsize=(10, 6))
    df_melted = df_metrics.melt(id_vars="Tool", var_name="Metric", value_name="Score")
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Tool", palette="viridis")
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_binary_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Safe", "Vulnerable"],
        yticklabels=["Safe", "Vulnerable"],
    )
    plt.xlabel("Predicted (Corgea)")
    plt.ylabel("Actual (DiverseVul)")
    plt.title("Binary Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def binary_analysis(data: list[dict]) -> None:
    all_binary_results = []

    y_true_labels = [ensure_list(entry.get("cwes", [])) for entry in data]
    y_true_binary = [1 if len(labels) > 0 else 0 for labels in y_true_labels]

    mlb = MultiLabelBinarizer()
    mlb.fit(y_true_labels)
    y_true_matrix = mlb.transform(y_true_labels)

    for tool_key, tool_name in TOOLS.items():
        logger.info("Analizowanie wynikow dla: %s", tool_name)

        y_pred_labels = [ensure_list(entry.get(tool_key, [])) for entry in data]
        y_pred_binary = [1 if len(labels) > 0 else 0 for labels in y_pred_labels]

        bin_metrics = calculate_binary_metrics(y_true_binary, y_pred_binary)
        bin_metrics["Tool"] = tool_name
        all_binary_results.append(bin_metrics)

        y_pred_matrix = mlb.transform(y_pred_labels)
        micro = f1_score(y_true_matrix, y_pred_matrix, average="micro", zero_division=0)
        macro = f1_score(y_true_matrix, y_pred_matrix, average="macro", zero_division=0)

        logger.info(
            "%s - Binary F1: %.4f | Micro F1: %.4f | Macro F1: %.4f",
            tool_name,
            bin_metrics["F1-Score"],
            micro,
            macro,
        )

    df_binary = pd.DataFrame(all_binary_results)

    df_binary.to_csv(PLOTS_DIR / "tools_comparison_metrics.csv", index=False)

    plot_comparison_bar(
        df_binary,
        "Binary Detection Comparison (Safe vs Vulnerable)",
        PLOTS_DIR / "binary_tools_comparison.png",
    )

    logger.info("Analiza zakonczona. Wykresy i raporty CSV w: %s", PLOTS_DIR)


def multilabel_analysis(data: list[dict]) -> None:
    # 1. Przygotowanie danych
    y_true = [ensure_list(entry.get("cwes", [])) for entry in data]

    # Inicjalizacja MLB na podstawie wszystkich etykiet (True + wszystkie narzędzia)
    all_possible_labels = set().union(*y_true)
    for key in TOOLS:
        for entry in data:
            all_possible_labels.update(ensure_list(entry.get(key, [])))

    mlb = MultiLabelBinarizer(classes=sorted(all_possible_labels))
    y_true_bin = mlb.fit_transform(y_true)

    # 2. Obliczanie metryk dla każdego narzędzia
    ml_results = []

    for key, name in TOOLS.items():
        y_pred = [ensure_list(entry.get(key, [])) for entry in data]
        y_pred_bin = mlb.transform(y_pred)

        # Jaccard Score (Intersection over Union) - kluczowe dla multi-label
        # Mierzy jak bardzo zestaw przewidzianych CWE pokrywa się z faktycznym
        jaccard = jaccard_score(
            y_true_bin, y_pred_bin, average="samples", zero_division=0
        )

        # F1 Micro i Macro
        f1_micro = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
        f1_macro = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)

        # Procentowy udział próbek, gdzie przewidziano PRZYNAJMNIEJ jedno poprawne CWE
        exact_match = np.all(y_true_bin == y_pred_bin, axis=1).mean()
        partial_match = 0
        for t, p in zip(y_true, y_pred, strict=False):
            if set(t).intersection(set(p)):
                partial_match += 1
        partial_match_pct = partial_match / len(data)

        ml_results.append(
            {
                "Tool": name,
                "Jaccard (Overlap)": jaccard,
                "F1 Micro": f1_micro,
                "F1 Macro": f1_macro,
                "Exact Match Ratio": exact_match,
                "Partial CWE Match": partial_match_pct,
            }
        )

    df_ml = pd.DataFrame(ml_results)
    logger.info("\n%s", df_ml.to_string(index=False))

    # --- WYKRES 1: Porównanie Metryk Multi-Label ---
    plt.figure(figsize=(12, 6))
    df_melt = df_ml.melt(id_vars="Tool", var_name="Metric", value_name="Score")
    sns.barplot(data=df_melt, x="Tool", y="Score", hue="Metric")
    plt.title("Porównanie skuteczności klasyfikacji Multi-Label (CWE)")
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "multi_label_metrics.png")

    # --- WYKRES 2: Analiza konkretnych CWE (Top 10 najczęstszych) ---
    # Sprawdzamy Recall dla najpopularniejszych CWE w DiverseVul
    true_cwe_counts = (
        pd.Series([cwe for sublist in y_true for cwe in sublist])
        .value_counts()
        .head(10)
    )
    top_10_cwes = true_cwe_counts.index.tolist()

    recall_per_cwe = []
    for cwe in top_10_cwes:
        cwe_idx = list(mlb.classes_).index(cwe)
        for key, name in TOOLS.items():
            y_pred_bin = mlb.transform(
                [ensure_list(entry.get(key, [])) for entry in data]
            )

            # Recall dla konkretnej klasy
            tp = np.sum((y_true_bin[:, cwe_idx] == 1) & (y_pred_bin[:, cwe_idx] == 1))
            fn = np.sum((y_true_bin[:, cwe_idx] == 1) & (y_pred_bin[:, cwe_idx] == 0))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            recall_per_cwe.append({"CWE": cwe, "Tool": name, "Recall": recall})

    df_cwe = pd.DataFrame(recall_per_cwe)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_cwe, x="CWE", y="Recall", hue="Tool")
    plt.title("Skuteczność wykrywania (Recall) dla 10 najczęstszych typów CWE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_cwe_recall_comparison.png")


def main():
    if not REPORT_JSON_FILE.exists():
        logger.error("Nie znaleziono pliku %s", REPORT_JSON_FILE)
        return
    PLOTS_DIR.mkdir(exist_ok=True)

    logger.info("Wczytywanie danych z %s", REPORT_JSON_FILE)
    try:
        with REPORT_JSON_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.exception("Nie udalo sie wczytac pliku JSON: %s", e)
        return
    binary_analysis(data)
    multilabel_analysis(data)


if __name__ == "__main__":
    main()
