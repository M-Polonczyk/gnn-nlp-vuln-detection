import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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


def ensure_list(x):
    if isinstance(x, str):
        return [x] if x else []
    return x if x is not None else []


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


def main():
    if not REPORT_JSON_FILE.exists():
        logger.error("Blad: Nie znaleziono pliku %s", REPORT_JSON_FILE)
        return

    PLOTS_DIR.mkdir(exist_ok=True)

    logger.info("Wczytywanie danych z %s", REPORT_JSON_FILE)
    try:
        with REPORT_JSON_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.exception("Nie udalo sie wczytac pliku JSON: %s", e)
        return

    y_true_labels = []
    y_pred_labels = []
    y_true_binary = []
    y_pred_binary = []

    for entry in data:
        true_cwes = ensure_list(entry.get("cwes", []))
        pred_cwes = ensure_list(entry.get("cwes_corgea", []))

        y_true_labels.append(true_cwes)
        y_pred_labels.append(pred_cwes)

        y_true_binary.append(1 if len(true_cwes) > 0 else 0)
        y_pred_binary.append(1 if len(pred_cwes) > 0 else 0)

    # --- ANALIZA 1: BINARNA ---
    logger.info("Obliczanie metryk binarnych (Vulnerable vs Safe)")

    bin_acc = accuracy_score(y_true_binary, y_pred_binary)
    bin_prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    bin_rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    bin_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    logger.info("Binary Accuracy: %s", bin_acc)
    logger.info("Binary Precision: %s", bin_prec)
    logger.info("Binary Recall: %s", bin_rec)
    logger.info("Binary F1-Score: %s", bin_f1)

    plot_binary_confusion_matrix(
        y_true_binary, y_pred_binary, PLOTS_DIR / "binary_confusion_matrix.png"
    )

    # --- ANALIZA 2: MULTI-LABEL ---
    logger.info("Obliczanie metryk multi-label dla konkretnych CWE")

    mlb = MultiLabelBinarizer()
    y_true_matrix = mlb.fit_transform(y_true_labels)
    y_pred_matrix = mlb.transform(y_pred_labels)

    classes = mlb.classes_
    logger.info("Liczba unikalnych klas CWE: %s", len(classes))

    micro_f1 = f1_score(y_true_matrix, y_pred_matrix, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true_matrix, y_pred_matrix, average="macro", zero_division=0)

    logger.info("Micro F1 Score: %s", micro_f1)
    logger.info("Macro F1 Score: %s", macro_f1)

    report_dict = classification_report(
        y_true_matrix,
        y_pred_matrix,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    report_csv_path = PLOTS_DIR / "metrics_per_cwe.csv"
    report_df.to_csv(report_csv_path)
    logger.info("Szczegolowy raport zapisano do: %s", report_csv_path)
    logger.info("Wykresy zapisano w katalogu: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
