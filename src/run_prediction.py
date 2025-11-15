import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.data_processing.graph_converter import DataclassToGraphConverter
from gnn_vuln_detection.models.factory import (
    GNNModelFactory,
)
from gnn_vuln_detection.utils import config_loader
from src.gnn_vuln_detection.dataset import DiverseVulDatasetLoader


def get_random_code_sample(dataset_config, labels):
    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )
    samples = diversevul_loader.load_dataset(labels)
    return random.choice(samples)


def main():
    dataset_config = config_loader.load_config("dataset_paths.yaml")
    model_params = config_loader.load_config("model_params.yaml")["gcn_multiclass"]
    cwes = config_loader.load_config("model_params.yaml")["vulnerabilities"]

    num_classes = model_params["num_classes"]
    cwe_to_index = {val["cwe_id"]: val["index"] for val in cwes}

    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )
    samples = diversevul_loader.load_dataset([val["cwe_id"] for val in cwes])

    for _ in range(10):
        code_sample = random.choice(samples)
        label_vec = [0] * num_classes
        if code_sample.cwe_ids:
            for cwe in code_sample.cwe_ids:
                if cwe in cwe_to_index:
                    label_vec[cwe_to_index[cwe]] = 1
        code_sample.cwe_ids_labeled = label_vec

        print("Code sample")
        print(f"{code_sample.cwe_ids=}")
        print(f"{code_sample.cwe_ids_labeled=}\n")

        converter = DataclassToGraphConverter()
        batch_graph = converter.code_sample_to_pyg_data(code_sample)

        input_dim = batch_graph.x.shape[1]

        model_params.pop("model_type", None)

        model = GNNModelFactory.create_model(
            model_type="gcn",
            input_dim=input_dim,
            num_classes=model_params["num_classes"],
            config=model_params,
        )
        model.load_state_dict(torch.load("cwe_detector.pth"))

        model.eval()
        with torch.no_grad():
            output = model(batch_graph.x, batch_graph.edge_index, batch_graph.batch)

        probs = torch.sigmoid(output)
        preds = (probs > 0.5).int()

        print("Probabilities:", probs)
        print("Predicted CWE labels:", preds)
        print("True CWE labels:", batch_graph.y)
        print("Predicted CWEs:")
        print([val["cwe_id"] for val in cwes if preds[0, val["index"]] == 1])


if __name__ == "__main__":
    main()
