"""Download CICIDS2017 from Kaggle, package it, and compute its checksum.

Run this ONCE, on a machine with Kaggle credentials, to produce the archive
that will be uploaded to Zenodo and attached to a GitHub Release. After
uploading, copy the printed SHA256 into data/download.py (EXPECTED_SHA256).

Requires Kaggle credentials: place kaggle.json in ~/.kaggle/ or set the
KAGGLE_USERNAME and KAGGLE_KEY environment variables. See data/README.md.

Usage:
    python scripts/package_dataset.py
"""

import hashlib
import os
import shutil
import sys
import zipfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from hif.config import KAGGLE_DATASET  # noqa: E402

DATA_DIR = os.path.join(_ROOT, "data", "cicids2017")
DIST_DIR = os.path.join(_ROOT, "dist")
ZIP_PATH = os.path.join(DIST_DIR, "cicids2017.zip")


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import kagglehub

    print("Downloading %s via kagglehub ..." % KAGGLE_DATASET)
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Downloaded to: %s" % path)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DIST_DIR, exist_ok=True)

    csv_files = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(".csv"):
                csv_files.append(os.path.join(root, name))
    if not csv_files:
        raise SystemExit("No CSV files found in the downloaded dataset.")

    print("Copying %d CSV files into %s" % (len(csv_files), DATA_DIR))
    for fp in csv_files:
        shutil.copy2(fp, os.path.join(DATA_DIR, os.path.basename(fp)))

    print("Writing archive %s" % ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(DATA_DIR)):
            if name.endswith(".csv"):
                zf.write(os.path.join(DATA_DIR, name),
                         arcname=os.path.join("cicids2017", name))

    digest = _sha256(ZIP_PATH)
    size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    sha_path = os.path.join(DIST_DIR, "cicids2017.zip.sha256")
    with open(sha_path, "w") as fh:
        fh.write("%s  cicids2017.zip\n" % digest)

    print("\nArchive ready.")
    print("  file   : %s" % ZIP_PATH)
    print("  size   : %.1f MB" % size_mb)
    print("  sha256 : %s" % digest)
    print("\nNext steps:")
    print("  1. Upload dist/cicids2017.zip to Zenodo and to a GitHub Release.")
    print("  2. Put the SHA256 above into EXPECTED_SHA256 in data/download.py.")
    print("  3. Put the Zenodo and Release URLs into data/download.py.")


if __name__ == "__main__":
    main()
