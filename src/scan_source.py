import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCAN_DIR = Path("data/code/diversevul").absolute()

SQ_RUN = f"""docker run \
--rm \
--network="host" \
-e SONAR_HOST_URL="http://localhost:9000"  \
-e SONAR_TOKEN={os.getenv("SONAR_TOKEN")} \
-v "{SCAN_DIR}:/usr/src" \
sonarsource/sonar-scanner-cli:12 \
-Dsonar.projectKey={os.getenv("SONAR_PROJECT")} \
-Dsonar.cfamily.compile-commands=compile_commands.json"""


def run(cmd: str) -> Any:
    try:
        return subprocess.run(shlex.split(cmd), check=False, capture_output=True)
    except subprocess.SubprocessError:
        return None


def start_scan(scanner: Literal["sq"]):
    ret = None
    if scanner == "sq":
        ret = run(SQ_RUN)
    if ret is None:
        logger.error("Failed to scan a resource")
    logger.info("Scan Results: %s", ret)


def main():
    scanners = ["sq"]
    logger.info(
        "Starting scan against %s directory using scanners: %s",
        SCAN_DIR,
        ", ".join(scanners),
    )
    for scanner in scanners:
        start_scan(scanner)


if __name__ == "__main__":
    main()
