"""Create a small, class-stratified sample of CICIDS2017 for the repository.

Reads the full dataset (archived local copy in data/cicids2017, or kagglehub),
draws a stratified sample by the ` Label` column, and writes it to
data/sample/cicids2017_sample.csv. This committed sample lets anyone run the
pipeline instantly without downloading the full dataset:

    python reproduce/run_comparison.py --data data/sample/cicids2017_sample.csv

Usage:
    python scripts/make_sample.py --rows 12000
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from hif import preprocessing  # noqa: E402
from hif.config import LABEL_COL, RANDOM_STATE  # noqa: E402

OUT = os.path.join(_ROOT, "data", "sample", "cicids2017_sample.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=12000, help="sample size")
    args = ap.parse_args()

    df = preprocessing.load_dataset()
    # Binary stratification key (BENIGN vs any attack), same as the pipeline.
    key = df[LABEL_COL].apply(lambda x: 0 if str(x).strip() == "BENIGN" else 1)
    frac = min(1.0, args.rows / len(df))
    _, sample = train_test_split(
        df, test_size=frac, stratify=key, random_state=RANDOM_STATE
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sample.to_csv(OUT, index=False)
    size_mb = os.path.getsize(OUT) / (1024 * 1024)
    print("Wrote %s" % OUT)
    print("  rows: %d   size: %.1f MB" % (len(sample), size_mb))
    print("  label distribution:")
    print(sample[LABEL_COL].apply(lambda x: "BENIGN" if str(x).strip() == "BENIGN"
                                  else "ATTACK").value_counts().to_string())


if __name__ == "__main__":
    main()
