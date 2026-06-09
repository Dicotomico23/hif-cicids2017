"""Reproduce the cleaned CICIDS2017 export from the raw CIC CSV files.

This documents, and makes reproducible, the cleaning that produced
``data/cicids2017_cleaned.csv`` (the canonical file hosted on the GitHub
Release). The released file remains the reference; this script regenerates it
from the original Canadian Institute for Cybersecurity CSVs to within ~0.03% of
the row count (small differences come from the exact de-duplication order).

Cleaning steps:
  1. Concatenate the eight raw CSV files and strip column-name whitespace.
  2. Drop 26 of the original feature columns: eight constant (zero-variance)
     columns, the duplicate ``Fwd Header Length.1``, columns whose values are
     identical/near-identical to a retained feature, and low-variance flag
     counters. 52 numeric features remain.
  3. Replace +/-inf with NaN and drop rows with any missing value.
  4. Drop exact duplicate flow records.
  5. Group the raw labels into the ``Attack Type`` categories used by the
     pipeline (any attack is later binarized to the anomalous class).

Usage:
  python data/clean_dataset.py --raw data/cicids2017 --out data/cicids2017_cleaned.csv
  python data/clean_dataset.py            # raw auto-downloaded via kagglehub
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# 26 columns removed from the raw 78-feature export.
DROP_COLUMNS = [
    # constant / zero-variance
    "Bwd PSH Flags", "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    # exact / near-duplicate of a retained feature
    "Fwd Header Length.1", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Total Backward Packets", "Total Length of Bwd Packets",
    # low-variance flag counters / redundant std
    "Fwd PSH Flags", "Fwd URG Flags",
    "SYN Flag Count", "RST Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Active Std", "Idle Std",
]

# Raw label -> grouped Attack Type (any attack is binarized downstream).
LABEL_GROUPS = {
    "BENIGN": "Normal Traffic",
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS", "Heartbleed": "DoS",
    "DDoS": "DDoS",
    "PortScan": "Port Scanning",
    "FTP-Patator": "Brute Force", "SSH-Patator": "Brute Force",
    "Bot": "Bots",
}


def _group_label(lbl):
    s = str(lbl).strip()
    if s in LABEL_GROUPS:
        return LABEL_GROUPS[s]
    if s.startswith("Web Attack"):  # the raw label carries a stray byte
        return "Web Attacks"
    return s  # e.g. the handful of Infiltration rows keep their own label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=None,
                    help="directory of raw CICIDS2017 CSVs "
                         "(auto-downloaded via kagglehub if omitted)")
    ap.add_argument("--out", default="data/cicids2017_cleaned.csv",
                    help="output CSV path")
    args = ap.parse_args()

    raw_dir = args.raw
    if raw_dir is None:
        import kagglehub
        raw_dir = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        sys.exit("No CSV files found in %s" % raw_dir)

    print("Reading %d raw files ..." % len(files))
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                   ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    print("Raw shape: %s" % str(df.shape))

    df["Attack Type"] = df["Label"].map(_group_label)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns] + ["Label"])

    feats = [c for c in df.columns if c != "Attack Type"]
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feats)
    df = df.drop_duplicates(subset=feats).reset_index(drop=True)

    print("Cleaned shape: %s (%d features)" % (str(df.shape), len(feats)))
    print(df["Attack Type"].value_counts().to_string())

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Wrote %s" % args.out)


if __name__ == "__main__":
    main()
