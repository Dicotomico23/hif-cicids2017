"""Download and verify the cleaned CICIDS2017 dataset.

The cleaned dataset (a single CSV: 2,520,751 flows, 52 features plus an
`Attack Type` column) is hosted as a zipped asset so the experiments stay
reproducible. This script downloads it, verifies the SHA256 checksum, and
extracts it to data/cicids2017_cleaned.csv, which the pipeline reads
automatically.

Sources are tried in order:
  1. a GitHub Release asset (primary)
  2. Zenodo (optional mirror, once configured)
  3. Kaggle (opt-in fallback for the raw dataset; needs credentials)

Usage:
    python data/download.py
    DATASET_URL=https://example/cicids2017_cleaned.zip python data/download.py
    ALLOW_KAGGLE=1 python data/download.py    # raw-dataset fallback
"""

import hashlib
import os
import sys
import urllib.request
import zipfile

RELEASE_URL = ("https://github.com/Dicotomico23/hif-cicids2017/releases/"
               "download/dataset-v1/cicids2017_cleaned.zip")
ZENODO_URL = ""  # optional mirror: https://zenodo.org/records/<id>/files/cicids2017_cleaned.zip
EXPECTED_SHA256 = "87ee289cd822407c06181cd04048de1f07f84d3f4912493b8a81ea610cea20d9"

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(_HERE, "cicids2017_cleaned.csv")
ZIP_PATH = os.path.join(_HERE, "cicids2017_cleaned.zip")
RAW_DIR = os.path.join(_HERE, "cicids2017")  # raw kaggle fallback target


def _already_present():
    return os.path.isfile(OUT_CSV)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(url, dest):
    print("  trying %s" % url)

    def _hook(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            sys.stdout.write("\r  downloading: %3d%%" % pct)
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _hook)
    sys.stdout.write("\n")


def _verify(path):
    if not EXPECTED_SHA256:
        print("  warning: EXPECTED_SHA256 not set, skipping checksum verification")
        return
    digest = _sha256(path)
    if digest != EXPECTED_SHA256:
        raise SystemExit(
            "Checksum mismatch.\n  expected %s\n  got      %s"
            % (EXPECTED_SHA256, digest)
        )
    print("  checksum OK")


def _extract_cleaned(zip_path):
    print("  extracting cleaned CSV to %s" % OUT_CSV)
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(member) as src, open(OUT_CSV, "wb") as dst:
            dst.write(src.read())


def _kaggle_allowed():
    return bool(os.environ.get("ALLOW_KAGGLE")) or ("--kaggle" in sys.argv)


def _from_kagglehub():
    import shutil

    import kagglehub

    sys.path.insert(0, os.path.join(_HERE, "..", "src"))
    from hif.config import KAGGLE_DATASET

    print("  falling back to kagglehub (%s)" % KAGGLE_DATASET)
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    os.makedirs(RAW_DIR, exist_ok=True)
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(".csv"):
                shutil.copy2(os.path.join(root, name), os.path.join(RAW_DIR, name))


def main():
    if _already_present():
        print("Dataset already present: %s" % OUT_CSV)
        return

    sources = [u for u in (os.environ.get("DATASET_URL"), RELEASE_URL, ZENODO_URL) if u]
    for url in sources:
        try:
            _download(url, ZIP_PATH)
            _verify(ZIP_PATH)
            _extract_cleaned(ZIP_PATH)
            os.remove(ZIP_PATH)
            print("Done.")
            return
        except Exception as exc:  # noqa: BLE001
            print("  failed: %s" % exc)

    if not _kaggle_allowed():
        raise SystemExit(
            "Could not download the cleaned dataset from the configured sources. "
            "Re-run with ALLOW_KAGGLE=1 to fetch the raw dataset from Kaggle "
            "instead, or set DATASET_URL to a reachable mirror."
        )

    print("Emergency fallback: fetching the raw dataset from Kaggle ...")
    try:
        _from_kagglehub()
        print("Done.")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Kaggle fallback failed: %s" % exc)


if __name__ == "__main__":
    main()
