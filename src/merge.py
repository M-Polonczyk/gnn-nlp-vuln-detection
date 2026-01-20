import csv
import json
import logging
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataMerger")

AIKIDO_JSON = Path("comparison/aikido_issues.json")
SEMGREP_CSV = Path("comparison/Semgrep_Code_Findings.csv")
OUTPUT_FILE = Path("comparison/output.json")


def get_hash_from_filename(filename: str) -> str:
    """Wyciąga BigHash z nazwy pliku (ostatnia część po podkreślniku)."""
    if not filename:
        return ""
    return Path(filename).stem.split("_")[-1]


def parse_semgrep_path(url: str) -> str:
    """Wyciąga nazwę pliku z URL Semgrepa."""
    match = re.search(r"_([0-9]+)\.c", url)
    return match.group(1) if match else ""


def main():
    merged_data = {}

    logger.info("Przetwarzanie wynikow Aikido...")
    if AIKIDO_JSON.exists():
        with AIKIDO_JSON.open(encoding="utf-8") as f:
            aikido_list = json.load(f)

            for finding in aikido_list:
                file_path = finding.get("affected_file", "")
                h = get_hash_from_filename(file_path)
                merged_data.setdefault(h, {"cwes_aikido": [], "cwes_semgrep": []})
                cwes = finding.get("cwe_classes", [])
                merged_data[h]["cwes_aikido"].extend(cwes)

    logger.info("Przetwarzanie wynikow Semgrep...")
    if SEMGREP_CSV.exists():
        with SEMGREP_CSV.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("Line Of Code Url", "")
                filename = parse_semgrep_path(url)
                h = get_hash_from_filename(filename)
                merged_data.setdefault(h, {"cwes_aikido": [], "cwes_semgrep": []})

                if h in merged_data:
                    # TODO: Extend this mapping as needed.
                    rule = row.get("Rule Name", "")
                    if "path-manipulation" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-22")
                    elif (
                        "unbounded-copy-to-stack-buffer" in rule
                        or "unvalidated-array-index" in rule
                    ):
                        merged_data[h]["cwes_semgrep"].append("CWE-120")
                    elif (
                        "function-use-after-free" in rule
                        or "local-variable-malloc-free" in rule
                        or "null-library-function" in rule
                    ):
                        merged_data[h]["cwes_semgrep"].append("CWE-416")
                    elif "double-free" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-415")
                    elif "info-leak-on-non-formated-string" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-532")
                    elif "correctness.sizeof-pointer-type.sizeof-pointer-type" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-467")
                    elif "insecure-hash" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-328")
                    elif "snprintf-source-size" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-787")
                    elif "alloc-strlen" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-131")
                    elif "tainted-allocation-size" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-125")
                    elif "world-writable-file" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-732")
                    elif "insecure-use-strtok-fn" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-676")
                    elif "command-injection-path" in rule:
                        merged_data[h]["cwes_semgrep"].append("CWE-78")

    # Clean duplicates
    for h in merged_data:
        merged_data[h]["cwes_aikido"] = list(set(merged_data[h]["cwes_aikido"]))
        merged_data[h]["cwes_semgrep"] = list(set(merged_data[h]["cwes_semgrep"]))

    # Export
    logger.info("Zapisywanie scalonych danych do %s", OUTPUT_FILE)
    with OUTPUT_FILE.open("r+", encoding="utf-8") as f:
        main_file = json.load(f)
        for i in range(len(main_file)):
            main_file[i].update(
                merged_data.get(
                    main_file[i]["id"], {"cwes_aikido": [], "cwes_semgrep": []}
                )
            )
        f.seek(0)
        f.write(json.dumps(main_file, indent=2))
        f.truncate()


if __name__ == "__main__":
    main()
