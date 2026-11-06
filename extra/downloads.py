"""
Dataset Downloads from kaggle
"""

import os
import sys
import time
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from kaggle.api.kaggle_api_extended import KaggleApi


DATASET = "routinemotiv/nomad-filtered"
OUTPUT_DIR = "./data"

PAGE_SIZE = 200
MAX_WORKERS = 4
MAX_RETRIES = 3

LABEL_PREFIX = "NOMAD-YOLO/labels/"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download_labels.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger("label-downloader")


def human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} TB"


def download_file(relative_path: str):
    destination = os.path.join(OUTPUT_DIR, relative_path)

    if os.path.isfile(destination) and os.path.getsize(destination) > 0:
        size = human_size(os.path.getsize(destination))

        logger.info(
            "[SKIP] %s | %s already exists",
            relative_path,
            size,
        )

        return "skipped", relative_path

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.time()

        logger.info(
            "[START] %s | Attempt %d/%d",
            relative_path,
            attempt,
            MAX_RETRIES,
        )

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    DATASET,
                    "-f",
                    relative_path,
                    "-p",
                    OUTPUT_DIR,
                    "--unzip",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
            )

            if not os.path.isfile(destination) or os.path.getsize(destination) == 0:
                raise RuntimeError("Downloaded file missing or empty")

            elapsed = time.time() - start
            size = os.path.getsize(destination)

            logger.info(
                "[DONE]  %s | %s | %.2fs",
                relative_path,
                human_size(size),
                elapsed,
            )

            return "downloaded", relative_path

        except Exception as exc:
            elapsed = time.time() - start

            error = (
                exc.stderr.strip()
                if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
                else str(exc)
            )

            logger.warning(
                "[RETRY] %s | Attempt %d/%d failed after %.2fs | %s",
                relative_path,
                attempt,
                MAX_RETRIES,
                elapsed,
                error,
            )

            if attempt == MAX_RETRIES:
                logger.error("[FAILED] %s", relative_path)

                return "failed", f"{relative_path}: {error}"

            time.sleep(2**attempt)

    return "failed", relative_path


def main():
    api = KaggleApi()
    api.authenticate()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Starting LABEL download")
    logger.info("Dataset: %s", DATASET)
    logger.info("Label prefix: %s", LABEL_PREFIX)
    logger.info("Output: %s", os.path.abspath(OUTPUT_DIR))
    logger.info("Workers: %d", MAX_WORKERS)
    logger.info("=" * 70)

    # ---------------------------------------------------------
    # First: collect ONLY label files
    # ---------------------------------------------------------

    label_files = []

    page_token = None

    while True:
        result = api.dataset_list_files(
            DATASET,
            page_token=page_token,
            page_size=PAGE_SIZE,
        )

        files = result.files

        page_labels = [
            file.name for file in files if file.name.startswith(LABEL_PREFIX)
        ]

        label_files.extend(page_labels)

        logger.info(
            "Scanned page: %d files | Found %d labels",
            len(files),
            len(page_labels),
        )

        page_token = result.next_page_token

        if not page_token:
            break

    logger.info("=" * 70)
    logger.info("Found %d label files", len(label_files))
    logger.info("=" * 70)

    if not label_files:
        logger.warning("No label files found!")
        return

    # ---------------------------------------------------------
    # Download labels concurrently
    # ---------------------------------------------------------

    downloaded = 0
    skipped = 0
    failed = []

    total_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, path): path for path in label_files}

        total = len(futures)

        for i, future in enumerate(as_completed(futures), 1):
            status, message = future.result()

            if status == "downloaded":
                downloaded += 1

            elif status == "skipped":
                skipped += 1

            else:
                failed.append(message)

            logger.info(
                "[PROGRESS] %d/%d | Downloaded: %d | Skipped: %d | Failed: %d",
                i,
                total,
                downloaded,
                skipped,
                len(failed),
            )

    elapsed = time.time() - total_start

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    logger.info("=" * 70)
    logger.info("LABEL DOWNLOAD FINISHED")
    logger.info("Total labels: %d", len(label_files))
    logger.info("Downloaded:   %d", downloaded)
    logger.info("Skipped:      %d", skipped)
    logger.info("Failed:       %d", len(failed))
    logger.info("Total time:   %.2f minutes", elapsed / 60)
    logger.info("Location:     %s", os.path.abspath(OUTPUT_DIR))
    logger.info("=" * 70)

    if failed:
        with open(
            "failed_label_downloads.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write("\n".join(failed))

        logger.error("Failed labels saved to failed_label_downloads.txt")


if __name__ == "__main__":
    main()
