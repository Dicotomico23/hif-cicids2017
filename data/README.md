# Data

The CICIDS2017 dataset is not committed to git: it is about 1.1 GB and a single
CSV exceeds GitHub's 100 MB per-file limit. To keep the experiments
reproducible even if the original Kaggle link disappears, the dataset is
preserved in two independent places and fetched by a script.

## Archived copies (redundant)

1. Zenodo: a permanent, citable archive with a DOI (primary source).
2. GitHub Release asset: a mirror attached to a release of this repository.

Both hold the same `cicids2017.zip`, verified by SHA256.

## Getting the data

```
python data/download.py
```

This tries Zenodo, then the GitHub Release, then Kaggle (if credentials are
available), verifies the checksum, and extracts the CSV files into
`data/cicids2017/`. Once present, `reproduce/run_comparison.py` reads from this
local copy automatically.

You can also point it at any mirror:

```
DATASET_URL=https://example/cicids2017.zip python data/download.py
```

## Recreating the archive (maintainers)

On a machine with Kaggle credentials (`~/.kaggle/kaggle.json`, or the
`KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables):

```
python scripts/package_dataset.py
```

This downloads the dataset from Kaggle, builds `dist/cicids2017.zip`, and
prints its SHA256. Upload that zip to Zenodo and to a GitHub Release, then put
the two URLs and the checksum into `data/download.py`
(`ZENODO_URL`, `RELEASE_URL`, `EXPECTED_SHA256`).

## Sources

- Kaggle mirror: `chethuhn/network-intrusion-dataset`
- Official: https://www.unb.ca/cic/datasets/ids-2017.html

Each row is a network flow described by CICFlowMeter features plus a ` Label`
column (the leading space is part of the original column name).
