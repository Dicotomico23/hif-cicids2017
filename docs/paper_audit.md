# Paper audit: pending, incomplete, and unjustified items

This is a verification of every `[PENDIENTE]` and `[POR REVISAR]` marker in
`main.tex`, cross-checked against the canonical code in this repository
(`src/hif/`, `reproduce/run_comparison.py`), which is the source of truth.

Scope decision (settled): the experiment is benign versus all attacks
(0 = BENIGN, 1 = any attack), NOT DoS only. Every "DoS" reference in the
methodology, metrics and limitations must be reworded to "anomalous (attack)".

The items are grouped by what is needed to close them.

---

## A. Resolvable now from the code (exact values to write)

| Paper location | Current text | Authoritative value (from code) |
|---|---|---|
| Scope (§4, §5, §5.1.1, §6, §8) | "DoS attacks only" | Benign vs all attacks. Label: 0 = BENIGN, 1 = any attack |
| Feature count (§4 vs §5.6) | 22 vs 20 | 22 (`TOP_K_FEATURES = 22`) |
| Feature method (§4 vs §5.6) | Information Gain vs SelectKBest | Mutual information (`mutual_info_classif`), top 22, estimated on a 10,000-row training subsample. Not SelectKBest |
| Original feature count | 78 | 78 (CICFlowMeter numeric columns) |
| Cleaning (§5.1.4) | "missing/infinite values imputed using the column mean"; "object features factorized" | Wrong. The code replaces inf with NaN and DROPS those rows (`dropna`). No mean imputation, no factorization |
| Split ratios + seed (§5.1.5) | PENDING | Stratified 70 / 15 / 15; `random_state = 42` |
| Borderline-SMOTE variant (§5.1.6) | PENDING | `BorderlineSMOTE(random_state=42, n_jobs=-1)`: kind = borderline-1, k_neighbors = 5, m_neighbors = 10, sampling_strategy = auto (minority oversampled to majority). Applied to the training set only |
| Scaling (§5.1.7) | StandardScaler fit on train | Correct |
| HIF aggregation (§5.2) | "averaged" | Arithmetic mean of the three member scores |
| HIF config table (Table hif-config) | PENDING | ntrees = 100 for all three; sample_size (psi) = 256, 512, 128; other params alpha1 = 0.5, alpha2 = 0.5; max depth = ceil(log2(psi)) |
| HIF library (§5.2) | PENDING | Custom implementation in this repo (`src/hif/forest.py`), Marteau Hybrid Isolation Forest formulation. Not sklearn IsolationForest. Cite Marteau |
| HIF supervision (§5.2) | "unsupervised, trained exclusively on benign" | Semi-supervised: trees are built from benign samples, but labelled anomalies are routed into the leaves to compute anomaly centroids |
| Threshold objective (Eq. threshold-objective) | linear w*P + (1-w)*R | Wrong. F-beta with beta = sqrt((1-w)/w), w = 0.9 |
| Algorithm 1 threshold step | "grid Theta of size N" | Wrong. Continuous bounded search (`scipy.optimize.minimize_scalar`, method bounded) over [min score, max score]. No grid, no N |
| Evaluation protocol (§5.3) | "baselines tuned, best validation F1" | No tuning. Baselines use fixed parameters and are trained once. HIF threshold is tuned on validation |
| Seeds / runs (§5.3, §8) | PENDING | Single global seed 42; single run |
| RF (§5.4) | PENDING | `RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)`; remaining defaults (max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', no class_weight) |
| NN/MLP (§5.4) | PENDING | `MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200, random_state=42, early_stopping=True)`; defaults: activation relu, solver adam, learning_rate_init 0.001, alpha 1e-4 |
| SVM (§5.4) | `sklearn.svm.SVC` | Wrong. `LinearSVC(random_state=42, max_iter=2000)` wrapped in `CalibratedClassifierCV`. Linear kernel, C = 1.0. No gamma |
| LOF (§5.4) | PENDING | `LocalOutlierFactor(n_neighbors=20, contamination=test anomaly fraction clipped to [0.001, 0.5], n_jobs=-1)`, novelty=False, used transductively (fit on train+test, read test rows) |
| Library versions (§5.7) | PENDING | numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, scikit-learn 1.4.2, imbalanced-learn 0.12.3, matplotlib 3.8.4, joblib 1.4.2, kagglehub 0.2.5. Optuna: not used (remove) |
| Reproducibility seed (§5.7) | PENDING | 42 |
| Code repository (§5.7 and Code availability) | PENDING (critical) | https://github.com/Dicotomico23/hif-cicids2017 (add Zenodo DOI when archived) |
| Metrics positive class (§6) | "DoS traffic" | "anomalous (attack) traffic" |
| Figure file names | fig_radar_metrics / fig_hif_confusion_matrix / fig_bar_balanced_acc / fig_bar_precision | Match the repo output exactly. The balanced-accuracy figure generator now exists (`metrics.plot_balanced_accuracy`) |

---

## B. Requires running the canonical pipeline (then fill from results/)

These are not author knowledge; they are produced by one run of
`reproduce/run_comparison.py` (which needs the dataset, ~1 GB). Per the agreed
decision, the paper numbers are updated to match this run.

- Absolute split sizes: train / validation / test row counts.
- Class distribution in the training set before and after Borderline-SMOTE
  (absolute and percentage).
- Table 5 (tab:performance): all metric values for HIF, RF, NN, SVM, LOF.
  The current values have uncertain provenance and must be regenerated.
- HIF confusion matrix counts. NOTE: the current numbers (419,012 benign /
  38,749 attack in test, an 8.5 percent attack rate) are NOT consistent with a
  70/15/15 split on the full all-attacks dataset (about 19.7 percent attacks);
  they match the old notebook's anomaly subsampling. They must be regenerated.
- Balanced accuracy values (the "about 85 percent" claim for HIF).
- LOF ROC AUC: the code passes continuous scores, so an exact 0.5000 is not
  expected; the real value will come from the run.
- Optional but recommended: the list of the 22 selected features with their
  mutual-information scores (table selected-features).
- Total pipeline runtime (depends on hardware).

---

## C. Requires the user (external facts, not in code or data)

- ORCID for the three authors.
- Funding statement (or confirm "Not applicable").
- Hardware of the run machine: CPU model and cores, RAM, GPU (if any).
- Operating system and exact Python version used for the reported run.
- Zenodo DOI for the dataset archive and (optionally) the code.
- Author contribution statement: currently lists only D.L.F.; confirm whether
  all three authors should be credited per role.
- Conflict of interest: confirm wording before submission.

---

## D. Claims in the text NOT supported by the canonical code (rewrite or remove)

These are the "unjustified" items. Each is currently asserted in the paper but
does not happen in the code.

1. DoS-only scope (the experiment is all-attacks).
2. 25 percent stratified subsample (the code uses the full cleaned dataset,
   then a 70/15/15 split; there is no 25 percent sampling step).
3. Mean imputation of missing values (the code drops rows instead).
4. Factorization of object-type features (the code keeps numeric columns only).
5. Optuna hyperparameter optimization, in §5.7 and the §7 discussion
   ("comprehensive hyperparameter optimization via Optuna"). No Optuna is used.
6. Baselines "tuned" on validation by best F1 (they use fixed parameters).
7. HIF described as fully unsupervised (it is semi-supervised).

---

## E. LaTeX and bibliography problems

- Broken cross-references: `\ref{subsec:sota-comparison}` (used in §2 and §8)
  and `\ref{subsec:precision-excellence}` (used in §5.2) point to labels that do
  not exist. Either create the labels or redirect to `sec:discussion`.
- Table 5 baseline row is missing the F1 and ROC AUC cells. From Talukder
  et al.: F1 about 0.960; ROC AUC not reported (use an em dash with a footnote).
- Bibliography: `\bibliography{references}` needs a `references.bib` containing
  every cited key (devi2019intrusion, pelletier2020evaluating, liu2022research,
  yulianto2019improving, talukder2024machine, ikram2021anomaly,
  primartha2017anomaly, soltani2022content, soltani2023multi,
  pekar2024evaluating, rosay2022network, cic2017dataset, sharafaldin2018detailed,
  gharib2016evaluation, damtew2023heterogeneous, kamalov2020feature,
  dimauro2021supervised, arreche2024xai, leevy2020survey, cantone2024cross).
  The old `custom.bib` uses different keys and will not resolve.
- Paste/typo corruptions to fix in the source: "Set eemplate", abstract
  "eved using metrics", "but s from outdated", "Recentudies", the table caption
  that reads "on{Comparative" (missing \caption), "CICIDS2017dataset",
  "contribun", table header "Otherameters", the algorithm line "\tehreshold",
  "emplontrusion detection", "\botruld{tabular}", "risk tolerce",
  "Hy Isolation Forest", "misclassid", "rensiveness", "three forestsa",
  "Conflictinterest", "wrote theript", and the broken MLP search-space bullet.
