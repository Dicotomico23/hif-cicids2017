"""Create a small, class-stratified sample of CICIDS2017 for the repository.

Reads the full dataset (by default the cleaned CSV in .external/, otherwise the
archived raw copy via the pipeline loader), draws a stratified sample by the
binary benign-vs-attack label, and writes data/sample/cicids2017_sample.csv.
This committed sample lets anyone run the pipeline instantly:

    python reproduce/run_comparison.py --data data/sample/cicids2017_sample.csv

Usage:
    python scripts/make_sample.py --rows 12000
    python scripts/make_sample.py --source path/to/cicids2017_cleaned.csv
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from hif import preprocessing  # noqa: E402
from hif.config import RANDOM_STATE  # noqa: E402

OUT = os.path.join(_ROOT, "data", "sample", "cicids2017_sample.csv")
DEFAULT_SOURCE = os.path.join(_ROOT, ".external", "cicids2017_cleaned.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=12000, help="sample size")
    ap.add_argument("--source", default=None,
                    help="source CSV (default: .external/cicids2017_cleaned.csv "
                         "if present, else the raw dataset via the loader)")
    args = ap.parse_args()

    source = args.source or (DEFAULT_SOURCE if os.path.exists(DEFAULT_SOURCE) else None)
    if source:
        print("Reading source: %s" % source)
        df = pd.read_csv(source, low_memory=False)
    else:
        df = preprocessing.load_dataset()

    label_col, benign = preprocessing.detect_label(df)
    key = df[label_col].apply(lambda x: 0 if str(x).strip() == benign else 1)
    frac = min(0.9999, args.rows / len(df))
    _, sample = train_test_split(
        df, test_size=frac, stratify=key, random_state=RANDOM_STATE
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sample.to_csv(OUT, index=False)
    size_mb = os.path.getsize(OUT) / (1024 * 1024)
    skey = sample[label_col].apply(lambda x: "BENIGN" if str(x).strip() == benign
                                   else "ATTACK")
    print("Wrote %s" % OUT)
    print("  rows: %d   size: %.1f MB   (label column: %s)"
          % (len(sample), size_mb, label_col))
    print(skey.value_counts().to_string())


if __name__ == "__main__":
    main()
