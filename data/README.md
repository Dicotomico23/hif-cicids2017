# Data

The CICIDS2017 dataset is not committed to git: it is about 1.1 GB and a single
CSV exceeds GitHub's 100 MB per-file limit. To keep the experiments
reproducible even if the original Kaggle link disappears, the dataset is
preserved in two independent places and fetched by a script.

## Archived copies (redundant)

1. Zenodo: a permanent, citable archive with a DOI (primary source).
2. GitHub Release asset: a mirror attached to a release of this repository.

Both hold the same `cicids2017.zip`, verified by SHA256.

## Dataset variant

The pipeline uses the cleaned CICIDS2017 export `cicids2017_cleaned.csv`
(2{,}520{,}751 flows, 52 features plus an `Attack Type` column with grouped
categories: Normal Traffic, DoS, DDoS, Port Scanning, Brute Force, Web Attacks,
Bots). The pipeline auto-detects this format (benign = `Normal Traffic`) as well
as the raw CICIDS2017 CSVs (benign = `BENIGN`); all attacks are collapsed into a
single anomalous class.

## Committed sample (instant testing)

A small, class-stratified sample is committed at
`data/sample/cicids2017_sample.csv` (about 12k rows, ~3 MB) so the pipeline can
be run immediately without any download:

```
python reproduce/run_comparison.py --data data/sample/cicids2017_sample.csv
```

It was produced from the full dataset with `scripts/make_sample.py` and
preserves the benign/attack class proportions. Use the full dataset (below) for
the reported results; use `--fraction` to run the full data on a smaller slice.

## Download the dataset directly (no script)

The full cleaned dataset is published as a single GitHub Release asset. You can
download it with a browser or any tool, no Python and no Kaggle account needed:

https://github.com/Dicotomico23/hif-cicids2017/releases/download/dataset-v1/cicids2017_cleaned.zip

The archive contains one file, `cicids2017_cleaned.csv`. Unzip it into this
`data/` directory and the pipeline reads it automatically:

```
curl -L -o cicids2017_cleaned.zip \
  https://github.com/Dicotomico23/hif-cicids2017/releases/download/dataset-v1/cicids2017_cleaned.zip
unzip cicids2017_cleaned.zip -d data/
```

SHA256 of the zip:
`87ee289cd822407c06181cd04048de1f07f84d3f4912493b8a81ea610cea20d9`

## Getting the full data (script, with checksum check)

```
python data/download.py
```

This downloads the cleaned dataset from the GitHub Release
(`dataset-v1`, asset `cicids2017_cleaned.zip`, ~196 MB), verifies its SHA256
checksum, and extracts it to `data/cicids2017_cleaned.csv`. No Kaggle account is
needed. Once present, `reproduce/run_comparison.py` reads it automatically.

You can also point it at any mirror, or fall back to the raw Kaggle dataset:

```
DATASET_URL=https://example/cicids2017_cleaned.zip python data/download.py
ALLOW_KAGGLE=1 python data/download.py    # raw dataset via kagglehub
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
