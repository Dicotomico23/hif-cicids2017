"""Download and verify the archived CICIDS2017 dataset.

The dataset is preserved in two places so the experiments remain reproducible
even if the original Kaggle link disappears:

  1. Zenodo  (permanent, citable DOI)        -- primary source
  2. GitHub Release asset                     -- mirror

This script tries the sources in order, verifies the SHA256 checksum, and
extracts the CSV files into data/cicids2017/. If both archives are unreachable
and kagglehub is installed with valid credentials, it falls back to Kaggle.

Usage:
    python data/download.py
    DATASET_URL=https://example/cicids2017.zip python data/download.py
"""

import hashlib
import os
import sys
import urllib.request
import zipfile

# Filled in after running scripts/package_dataset.py and uploading the archive.
ZENODO_URL = ""   # e.g. https://zenodo.org/records/<id>/files/cicids2017.zip
RELEASE_URL = ""  # e.g. https://github.com/<owner>/<repo>/releases/download/<tag>/cicids2017.zip
EXPECTED_SHA256 = ""  # paste the checksum printed by package_dataset.py

_HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(_HERE, "cicids2017")
ZIP_PATH = os.path.join(_HERE, "cicids2017.zip")


def _already_present():
    return os.path.isdir(TARGET_DIR) and any(
        n.endswith(".csv") for n in os.listdir(TARGET_DIR)
    )


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


def _extract(path):
    print("  extracting into %s" % TARGET_DIR)
    os.makedirs(TARGET_DIR, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            if member.endswith(".csv"):
                data = zf.read(member)
                with open(os.path.join(TARGET_DIR, os.path.basename(member)), "wb") as fh:
                    fh.write(data)


def _from_kagglehub():
    import shutil

    import kagglehub

    sys.path.insert(0, os.path.join(_HERE, "..", "src"))
    from hif.config import KAGGLE_DATASET

    print("  falling back to kagglehub (%s)" % KAGGLE_DATASET)
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    os.makedirs(TARGET_DIR, exist_ok=True)
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(".csv"):
                shutil.copy2(os.path.join(root, name),
                             os.path.join(TARGET_DIR, name))


def main():
    if _already_present():
        print("Dataset already present in %s" % TARGET_DIR)
        return

    sources = [u for u in (os.environ.get("DATASET_URL"), ZENODO_URL, RELEASE_URL) if u]

    for url in sources:
        try:
            _download(url, ZIP_PATH)
            _verify(ZIP_PATH)
            _extract(ZIP_PATH)
            os.remove(ZIP_PATH)
            print("Done.")
            return
        except Exception as exc:  # noqa: BLE001
            print("  failed: %s" % exc)

    print("Archive sources unavailable; trying Kaggle ...")
    try:
        _from_kagglehub()
        print("Done.")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Could not obtain the dataset. Set ZENODO_URL/RELEASE_URL in this "
            "file, or configure Kaggle credentials. Details: %s" % exc
        )


if __name__ == "__main__":
    main()
