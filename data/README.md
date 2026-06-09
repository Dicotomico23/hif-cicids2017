# Data

The CICIDS2017 dataset is not stored in this repository. It is downloaded
automatically the first time you run the pipeline.

## How the download works

`reproduce/run_comparison.py` calls `kagglehub.dataset_download` on the
dataset `chethuhn/network-intrusion-dataset`, a mirror of the CICIDS2017
MachineLearning CSV export produced by the Canadian Institute for
Cybersecurity. kagglehub caches the files under `~/.cache/kagglehub`, so the
download happens only once.

To use kagglehub you need Kaggle credentials. Either:

- log in once with the Kaggle CLI, or
- place a `kaggle.json` API token in `~/.kaggle/kaggle.json`.

See https://www.kaggle.com/docs/api for details.

## What the pipeline reads

Every `.csv` file in the downloaded folder is concatenated. Each row is one
network flow described by the CICFlowMeter features plus a ` Label` column
(the leading space is part of the original column name).

## Official source

The original dataset and its documentation are available from the Canadian
Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html
