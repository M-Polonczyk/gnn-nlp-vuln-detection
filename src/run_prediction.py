import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.data_processing.graph_converter import DataclassToGraphConverter
from gnn_vuln_detection.dataset import DiverseVulDatasetLoader
from gnn_vuln_detection.models.factory import (
    GNNModelFactory,
)
from gnn_vuln_detection.utils import config_loader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_config = config_loader.load_config("dataset_paths.yaml")
    model_params = config_loader.load_config("model_params.yaml")["gcn_multiclass"]
    cwes = config_loader.load_config("model_params.yaml")["vulnerabilities"]

    num_classes = model_params["num_classes"]
    cwe_to_index = {val["cwe_id"]: val["index"] for val in cwes}
    index_to_cwe = {v: k for k, v in cwe_to_index.items()}

    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )
    converter = DataclassToGraphConverter()
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
        model.to(device)

        y_true, y_probs, y_labels = model.evaluate([batch_graph], device)
        print("y_true:", y_true)
        print("y_probs:", y_probs)
        print("y_labels:", y_labels)

        with torch.no_grad():
            output = model(batch_graph.x, batch_graph.edge_index, batch_graph.batch)

        print("Logits:", output)
        print("Sigmoid:", torch.sigmoid(output))

        probs = torch.sigmoid(output).squeeze(0)
        preds = (probs > 0.5).int()

        print("Probabilities:", probs)
        print("Predicted CWE labels:", preds)
        print("True CWE labels:", batch_graph.y)
        cwes_predicted = [val["cwe_id"] for val in cwes if preds[val["index"]] == 1]
        if cwes_predicted:
            print("Predicted CWEs:")
            print(", ".join(cwes_predicted))
            exit()

        best_idx = probs.argmax().item()
        best_prob = probs[int(best_idx)].item()
        best_cwe = index_to_cwe[best_idx]
        print("Best prediction:")
        print(f"\tCWE: {best_cwe}")
        print(f"\tindex: {best_idx}")
        print(f"\tprobability: {best_prob:.4f}")


if __name__ == "__main__":
    main()
