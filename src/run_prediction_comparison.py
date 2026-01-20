import json
import logging
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

# Dodaj ścieżkę do modułów GNN (zgodnie z Twoją strukturą)
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_vuln_detection.models.factory import GNNModelFactory
from gnn_vuln_detection.utils import config_loader, file_loader

# --- KONFIGURACJA ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GNN_Predictor")

OUTPUT_JSON_FILE = Path("comparison/output.json")
MODEL_CHECKPOINT = "checkpoints/cwe_detector.pth"
DATASET_PATH = "data/processed/diversevul_code_samples_with_meta.pt"
THRESHOLDS_PATH = "checkpoints/optimal_thresholds.csv"
CONFIG_PATH = "model_params.yaml"


def get_hash_from_filename(filename: str) -> str:
    """Wyciąga hash ID z nazwy pliku (ostatni człon po podkreślniku)."""
    # Zakładamy format: Project_GitHash_BigHash.c
    stem = Path(filename).stem
    parts = stem.split("_")
    return parts[-1] if parts else ""


def load_trained_model(input_dim, num_classes, device, config):
    """Tworzy instancję modelu i ładuje wagi."""
    model_config = config.copy()
    model_config.pop("model_type", None)

    model = GNNModelFactory.create_model(
        model_type="gcn",  # lub pobierz z configu
        input_dim=input_dim,
        num_classes=num_classes,
        config=model_config,
    )

    if Path(MODEL_CHECKPOINT).exists():
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
        logger.info(f"Zaladowano wagi modelu z {MODEL_CHECKPOINT}")
    else:
        logger.error(f"Nie znaleziono pliku modelu: {MODEL_CHECKPOINT}")
        sys.exit(1)

    model.to(device)
    model.eval()
    return model


def main():
    # 1. Wczytaj istniejący output.json
    if not OUTPUT_JSON_FILE.exists():
        logger.error(
            f"Nie znaleziono pliku {OUTPUT_JSON_FILE}. Uruchom najpierw merge_all_tools.py"
        )
        return

    logger.info(f"Wczytywanie {OUTPUT_JSON_FILE}...")
    with OUTPUT_JSON_FILE.open(encoding="utf-8") as f:
        json_data = json.load(f)

    # Mapowanie ID -> Obiekt (dla szybkiej edycji)
    results_map = {entry["id"]: entry for entry in json_data}

    # 2. Przygotuj środowisko GNN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Uzywane urzadzenie: {device}")

    # Wczytaj konfigurację
    full_config = config_loader.load_config(CONFIG_PATH)
    gnn_params = full_config["gcn_multiclass"]
    cwes_list = full_config["vulnerabilities"]

    # Mapowanie Index -> Nazwa CWE
    index_to_cwe = {val["index"]: val["cwe_id"] for val in cwes_list}
    num_classes = gnn_params["num_classes"]

    # Wczytaj progi (Thresholds)
    if Path(THRESHOLDS_PATH).exists():
        thresholds_str = file_loader.load_file(THRESHOLDS_PATH)
        thresholds = [float(t) for t in thresholds_str.split(",")]
        thresholds_tensor = torch.tensor(thresholds, device=device)
        logger.info("Zaladowano optymalne progi (thresholds).")
    else:
        thresholds_tensor = torch.tensor([0.5] * num_classes, device=device)
        logger.warning("Brak pliku z progami. Uzywam domyslnego 0.5")

    # 3. Wczytaj Dataset
    logger.info(f"Wczytywanie datasetu {DATASET_PATH}...")
    dataset = torch.load(DATASET_PATH, weights_only=False)

    # Ważne: Shuffle=False, aby zachować kolejność (choć polegamy na filename)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Pobierz wymiar wejściowy z pierwszego grafu
    input_dim = dataset[0].x.shape[1]

    # 4. Inicjalizacja Modelu
    model = load_trained_model(input_dim, num_classes, device, gnn_params)
    model.eval()
    # 5. Pętla Predykcji
    logger.info("Rozpoczynanie predykcji...")
    updated_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(logits)

            # Aplikacja progów (per klasa)
            preds = (probs > thresholds_tensor).int()

            # Obsługa metadanych (filename/ID)
            # PyG Data object przechowuje niestandardowe atrybuty.
            # Jeśli 'filename' jest listą (w batchu), używamy jej.
            if not hasattr(batch, "code_sample_id"):
                logger.warning(
                    "Batch danych nie posiada atrybutu 'code_sample_id'. Nie mozna zmapowac wynikow!"
                )
                continue

            ids = batch.code_sample_id

            # Iteracja po próbkach w batchu
            for i, file_id in enumerate(ids):
                # Dekodowanie CWE
                sample_preds = preds[i]
                found_cwes = []
                for idx, is_active in enumerate(sample_preds):
                    if is_active == 1:
                        cwe_name = index_to_cwe.get(idx, f"Unknown-{idx}")
                        found_cwes.append(cwe_name)

                # Aktualizacja JSON
                if file_id in results_map:
                    results_map[file_id]["cwes_gnn"] = found_cwes
                    updated_count += 1
                else:
                    # Pliki, których nie ma w raporcie JSON
                    msg = f"Nie znaleziono ID {file_id} w wynikach JSON."
                    raise ValueError(msg)

    logger.info("Zapisywanie scalonych danych do %s", OUTPUT_JSON_FILE)
    with OUTPUT_JSON_FILE.open("r+", encoding="utf-8") as f:
        main_file = json.load(f)
        for i in range(len(main_file)):
            main_file[i].update(results_map.get(main_file[i]["id"], {"cwes_gnn": []}))
        f.seek(0)
        f.write(json.dumps(main_file, indent=2))
        f.truncate()


if __name__ == "__main__":
    main()
