from gnn_vuln_detection.dataset.loaders import DiverseVulDatasetLoader
from gnn_vuln_detection.utils import config_loader

from .nlp_analysis import run as run_nlp_analysis
from .prediction import run as run_prediction


def run() -> None:
    code_samples = []

    dataset_config = config_loader.load_config("dataset_paths.yaml")
    cwes = config_loader.load_config("model_params.yaml")["vulnerabilities"]
    diversevul_loader = DiverseVulDatasetLoader(
        dataset_path=dataset_config["diversevul"]["dataset_path"],
    )
    code_samples = diversevul_loader.load_dataset([val["cwe_id"] for val in cwes])
    run_prediction(code_samples, cwes)
    # run_nlp_analysis()
