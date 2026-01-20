import gc
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn_vuln_detection.code_representation.feature_extractor import CodeGraphProcessor
from gnn_vuln_detection.data_processing.graph_converter import DataclassToGraphConverter
from gnn_vuln_detection.dataset.loaders import DiverseVulDatasetLoader
from gnn_vuln_detection.utils import config_loader


def main():
    dataset_config = config_loader.load_config("dataset_paths.yaml")
    model_params = config_loader.load_config("model_params.yaml")

    cwe_to_index = {
        val["cwe_id"]: val["index"] for val in model_params["vulnerabilities"]
    }
    num_classes = model_params["gcn_multiclass"]["num_classes"]

    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )

    converter = DataclassToGraphConverter()
    ast_parser = converter.ast_parser
    samples = diversevul_loader.load_dataset(list(cwe_to_index.keys()))
    np.random.shuffle(samples)
    samples = samples[: len(samples) // 36]

    for i, s in enumerate(tqdm(samples, desc="Building Label Matrix")):
        label_vec = [0] * num_classes
        if s.cwe_ids:
            for cwe in s.cwe_ids:
                if cwe in cwe_to_index:
                    label_vec[cwe_to_index[cwe]] = 1
        samples[i].cwe_ids_labeled = label_vec

    pyg_data_list = []

    for i in tqdm(range(len(samples)), desc="Saving code samples to files"):
        sample_code = ast_parser.cleanup_code(samples[i].code)
        with Path(
            f"data/code/diversevul/{samples[i].metadata.project}_{samples[i].metadata.commit_id}_{samples[i].id}.c"
        ).open("w") as f:
            f.write(sample_code)

        ast_root = ast_parser.parse_code_to_ast(sample_code)
        samples[i].graph = converter.ast_converter.ast_to_networkx(ast_root)
        samples[i].code = None  # Free up memory
    del converter
    del ast_parser
    gc.collect()
    processor = CodeGraphProcessor(
        node_dim=model_params["gcn_multiclass"]["hidden_dim"]
    )
    processor.fit([s.graph for s in samples])

    for s in tqdm(samples, desc="Processing samples"):
        features = processor.process(s.graph)
        x = torch.tensor(features.node_features, dtype=torch.float)
        edge_index = torch.tensor(features.edge_index, dtype=torch.long)
        y = torch.tensor(s.cwe_ids_labeled, dtype=torch.float32).unsqueeze(0)
        data_dict = {
            "x": x,
            "y": y,
            "edge_index": edge_index,
            "edge_features": torch.tensor(features.edge_features, dtype=torch.float)
            if features.edge_features is not None
            else None,
        }
        data = Data(**data_dict)
        data.code_sample_id = str(s.id)  # Otherwise torch sets it to long
        pyg_data_list.append(data)

    torch.save(pyg_data_list, "data/processed/diversevul_code_samples.pt")


if __name__ == "__main__":
    main()
