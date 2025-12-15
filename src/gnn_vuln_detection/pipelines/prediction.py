import logging
from typing import Literal

import torch
from torch_geometric.data import Data

from gnn_vuln_detection.code_representation.code_representation import CodeSample
from gnn_vuln_detection.data_processing.graph_converter import (
    DataclassToGraphConverter,
)
from gnn_vuln_detection.dataset.loaders import DiverseVulDatasetLoader
from gnn_vuln_detection.models.factory import (
    GNNModelFactory,
)
from gnn_vuln_detection.utils import config_loader

type Device = Literal["cpu", "cuda"]
PREDICTION_THRESHOLD = 0.5


def predict(batch_graph: Data, device: Device = "cpu") -> torch.Tensor:
    model_params = config_loader.load_config("model_params.yaml")["gcn_multiclass"]
    input_dim = batch_graph.x.shape[1]

    model_params.pop("model_type", None)

    model = GNNModelFactory.create_model(
        model_type="gcn",
        input_dim=input_dim,
        num_classes=model_params["num_classes"],
        config=model_params,
    )
    model.load_state_dict(torch.load("checkpoints/cwe_detector.pth"))
    model.to(device)

    y_true, y_probs, y_labels = model.evaluate([batch_graph], device)
    logging.info("y_true: %s", y_true)
    logging.info("y_probs: %s", y_probs)
    logging.info("y_labels: %s", y_labels)

    with torch.no_grad():
        return model(batch_graph.x, batch_graph.edge_index, batch_graph.batch)


def convert_to_batch_graphs(
    samples: list[CodeSample],
    num_classes: int,
    cwe_to_index: dict,
) -> list[Data]:
    converter = DataclassToGraphConverter()
    graphs = []
    for sample in samples:
        label_vec = [0] * num_classes
        if sample.cwe_ids:
            for cwe in sample.cwe_ids:
                if cwe in cwe_to_index:
                    label_vec[cwe_to_index[cwe]] = 1
        sample.cwe_ids_labeled = label_vec
        graphs.append(converter.code_sample_to_pyg_data(sample))
    return graphs


def run(code_samples: list[CodeSample], cwes: list[dict]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_params = config_loader.load_config("model_params.yaml")["gcn_multiclass"]

    num_classes = model_params["num_classes"]
    cwe_to_index = {val["cwe_id"]: val["index"] for val in cwes}
    index_to_cwe = {v: k for k, v in cwe_to_index.items()}

    samples = convert_to_batch_graphs(code_samples, num_classes, cwe_to_index)
    for i in range(10):
        batch_graph = samples[i].to(device)
        outputs = predict(batch_graph, device=device)
        predicted_probs = torch.sigmoid(outputs).cpu().numpy()[0]
        predicted_indices = [
            idx
            for idx, prob in enumerate(predicted_probs)
            if prob >= PREDICTION_THRESHOLD
        ]
        predicted_cwes = [index_to_cwe[idx] for idx in predicted_indices]

        logging.info("Code sample")
        logging.info("True CWEs: %s", code_samples[i].cwe_ids)
        logging.info("Predicted CWEs: %s", predicted_cwes)
        logging.info("Predicted probabilities: %s\n", predicted_probs)


def main():
    code_samples = []

    dataset_config = config_loader.load_config("dataset_paths.yaml")
    cwes = config_loader.load_config("model_params.yaml")["vulnerabilities"]
    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )
    code_samples = diversevul_loader.load_dataset([val["cwe_id"] for val in cwes])
    run(code_samples, cwes)


if __name__ == "__main__":
    main()
