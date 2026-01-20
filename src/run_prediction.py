import logging
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.models.factory import (
    GNNModelFactory,
)
from gnn_vuln_detection.training import metrics
from gnn_vuln_detection.utils import config_loader, file_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
THRESHOLD = 0.6
MODEL_PATH = "checkpoints/cwe_detector.pth"


def load_model(input_dim, device):
    model_params = config_loader.load_config("model_params.yaml")["gcn_multiclass"]
    model_params.pop("model_type", None)

    model = GNNModelFactory.create_model(
        model_type="gcn",
        input_dim=input_dim or model_params["input_dim"],  # or infer once
        num_classes=model_params["num_classes"],
        config=model_params,
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(batch_graph: Data, device="cpu") -> torch.Tensor:
    model_params = config_loader.load_config("model_params.yaml")["gcn_multiclass"]
    input_dim = batch_graph.x.shape[1]

    model_params.pop("model_type", None)

    model = GNNModelFactory.create_model(
        model_type="gcn",
        input_dim=input_dim,
        num_classes=model_params["num_classes"],
        config=model_params,
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    y_true, y_probs, y_labels = model.evaluate([batch_graph], device)
    logging.info("y_true: %s", y_true)
    logging.info("y_probs: %s", y_probs)
    logging.info("y_labels: %s", y_labels)

    with torch.no_grad():
        return model(batch_graph.x, batch_graph.edge_index, batch_graph.batch)


def load_dataset(dataset_type="test") -> DataLoader:
    data = torch.load("data/processed/diversevul_code_samples.pt", weights_only=False)
    # data = torch.load(
    #     f"data/processed/{dataset_type}-diversevul-c.pt", weights_only=False
    # )
    if isinstance(data, DataLoader):
        return data
    return DataLoader(
        data,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_params = config_loader.load_config("model_params.yaml")["gcn_multiclass"]
    cwes = config_loader.load_config("model_params.yaml")["vulnerabilities"]

    num_classes = model_params["num_classes"]
    cwe_to_index = {val["cwe_id"]: val["index"] for val in cwes}
    index_to_cwe = {v: k for k, v in cwe_to_index.items()}
    thresholds = [
        float(t)
        for t in file_loader.load_file("checkpoints/optimal_thresholds.csv").split(",")
    ]

    test_samples = load_dataset()
    model = load_model(input_dim=test_samples.dataset[0].x.shape[1], device=device)

    model.label_threshold = 0.5

    y_true, y_pred_probs, y_pred_labels = model.evaluate(test_samples, device)

    logging.info("y_true: %s", y_true)
    logging.info("y_probs: %s", y_pred_probs)
    logging.info("y_labels: %s", y_pred_labels)
    logging.info("Beta: %s", 1.5)
    calculated_metrics = metrics.calculate_metrics(
        y_true, y_pred_probs, y_pred_labels, 1.5, "macro"
    )
    logging.info("Calculated metrics: %s", calculated_metrics)
    # metrics.save_metrics_to_csv(calculated_metrics, "checkpoints/metrics.csv")

    for i in range(num_classes):
        class_cwe = index_to_cwe[i]
        class_true = [y_true[j][i] for j in range(len(y_true))]
        class_preds = [y_pred_labels[j][i] for j in range(len(y_pred_labels))]
        class_f1 = f1_score(class_true, class_preds)
        logging.info("F1 score for CWE %s (index %d): %.4f", class_cwe, i, class_f1)

    # for y_t, pred in zip(y_true, y_pred_labels, strict=False):
    #     cwes_predicted = [val["cwe_id"] for val in cwes if pred[val["index"]] == 1]
    #     if cwes_predicted:
    #         logging.info("Predicted CWEs: %s", ", ".join(cwes_predicted))
    #         logging.info(
    #             "Actual CWEs: %s",
    #             ", ".join([index_to_cwe[i] for i, v in enumerate(y_t) if v == 1]),
    #         )

    metrics.plot_confusion_matrix(y_true, y_pred_labels, labels=cwe_to_index.keys())


if __name__ == "__main__":
    main()
