"""Hybrid Isolation Forest (HIF) following Marteau's formulation.

A single HIF combines three signals into one anomaly score:

    s   isolation depth        standard Isolation Forest path length
    sc  centroid distance      Euclidean distance to the normal-data centroid
    sa  anomaly proximity      ratio of normal-centroid distance to the
                               distance to the nearest known-anomaly centroid

The final score is:

    shif = a2 * (a1 * s + (1 - a1) * sc) + (1 - a2) * sa

with a1 = a2 = 0.5 in this study. The model is trained on benign samples
(to learn normal behaviour) and then routes labelled anomalies into the
leaves so each leaf can store an anomaly centroid; this is what makes it
semi-supervised rather than a plain Isolation Forest.

The tree builder and scorers are iterative on purpose: a recursive version
hits Python's recursion limit on deep trees built from large subsamples.
"""

import math

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import precision_score, recall_score

from .config import RANDOM_STATE


class exNode:
    """External (leaf) node of a HIF tree.

    Attributes:
        c_lenS: precomputed c(|S|) path-length correction for the leaf.
        CS:     centroid of the normal samples that landed in this leaf.
        Xa:     anomaly vectors routed here after training.
        Ca:     centroid of those anomaly vectors (set by compute_anomaly_centroids).
    """

    def __init__(self, c_lenS, CS):
        self.c_lenS = c_lenS
        self.CS = CS
        self.Xa = []
        self.Ca = None


class inNode:
    """Internal (split) node of a HIF tree."""

    def __init__(self, left, right, splitDim, splitVal):
        self.left = left
        self.right = right
        self.splitDim = splitDim
        self.splitVal = splitVal


def _c(size):
    """Expected path length of an unsuccessful BST search (Liu et al., 2008)."""
    if size <= 1:
        return 0.0
    return 2.0 * (np.log(size - 1) + 0.5772156649) - 2.0 * (size - 1) / size


def _build_tree(S, lmax, rng):
    """Build one isolation tree iteratively to avoid the recursion limit."""
    stack = [(S, 0, None, False)]  # (data, depth, parent_inNode, is_left)
    root = None

    while stack:
        S_sub, depth, parent, is_left = stack.pop()
        n = len(S_sub)

        if depth >= lmax or n <= 1:
            centroid = S_sub.mean(axis=0) if n > 0 else None
            node = exNode(c_lenS=_c(n), CS=centroid)
        else:
            q = rng.integers(0, S_sub.shape[1])
            lo, hi = S_sub[:, q].min(), S_sub[:, q].max()
            if lo == hi:
                # All values identical on this dimension: make it a leaf.
                centroid = S_sub.mean(axis=0)
                node = exNode(c_lenS=_c(n), CS=centroid)
            else:
                p = rng.uniform(lo, hi)
                mask = S_sub[:, q] < p
                node = inNode(left=None, right=None, splitDim=q, splitVal=p)
                # Push right before left so the left child is processed first.
                stack.append((S_sub[~mask], depth + 1, node, False))
                stack.append((S_sub[mask], depth + 1, node, True))

        if parent is None:
            root = node
        elif is_left:
            parent.left = node
        else:
            parent.right = node

    return root


def _score_sample(x, tree):
    """Traverse one tree and return (h_x, delta_x, has_c, delta_ax, has_a).

    has_c / has_a flag whether the leaf has a normal / anomaly centroid, so the
    cross-tree averages can skip leaves that have none (as in Marteau's HIF)
    instead of averaging in spurious zeros.
    """
    e = 0
    node = tree
    while not isinstance(node, exNode):
        if x[node.splitDim] < node.splitVal:
            node = node.left
        else:
            node = node.right
        e += 1

    h_x = e + node.c_lenS
    if node.CS is not None:
        delta_x, has_c = float(np.linalg.norm(x - node.CS)), 1.0
    else:
        delta_x, has_c = 0.0, 0.0
    if node.Ca is not None:
        delta_ax, has_a = float(np.linalg.norm(x - node.Ca)), 1.0
    else:
        delta_ax, has_a = 0.0, 0.0
    return h_x, delta_x, has_c, delta_ax, has_a


def _score_one_tree(tree, X):
    """Score every sample against a single tree (used by joblib workers)."""
    n = len(X)
    h = np.zeros(n)
    dx = np.zeros(n)
    cx = np.zeros(n)
    da = np.zeros(n)
    ca = np.zeros(n)
    for i in range(n):
        h[i], dx[i], cx[i], da[i], ca[i] = _score_sample(X[i], tree)
    return h, dx, cx, da, ca


def _route_anomaly(x, tree):
    """Route one anomaly vector to its leaf and store it there."""
    node = tree
    while not isinstance(node, exNode):
        if x[node.splitDim] < node.splitVal:
            node = node.left
        else:
            node = node.right
    node.Xa.append(x)


def _compute_Ca(tree):
    """Compute the anomaly centroid Ca at every leaf of a tree."""
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, exNode):
            node.Ca = np.mean(node.Xa, axis=0) if node.Xa else None
        else:
            stack.append(node.left)
            stack.append(node.right)


class HybridIsolationForest:
    """A single Hybrid Isolation Forest.

    Args:
        t:   number of trees in the forest.
        psi: subsample size drawn (without replacement) to build each tree.
    """

    def __init__(self, t=100, psi=256):
        self.t = t
        self.psi = psi
        self.lmax = math.ceil(math.log2(psi))
        self.forest = []
        self.alpha1 = 0.5
        self.alpha2 = 0.5
        # Expected path length normalizer c(psi), used to turn the mean path
        # length into an isolation score 2^(-h/c) (high for anomalies), as in
        # Marteau's original HIF.
        self._c = _c(psi)
        # Normalization bounds for (h, dx, ratio), fitted once on a reference
        # set by calibrate(). Kept fixed afterwards so the threshold chosen on
        # the validation set transfers correctly to the test set. None means
        # "normalize per input batch" (used before calibration).
        self._norm = None

    def fit(self, X_normal, X_anomalies=None, y_anomalies=None):
        """Train on benign data, then route labelled anomalies into the trees."""
        n = X_normal.shape[0]
        if self.psi > n:
            raise ValueError(
                "psi (%d) cannot exceed the number of benign training samples (%d)."
                % (self.psi, n)
            )

        rng = np.random.default_rng(RANDOM_STATE)

        def _build_one(seed):
            local_rng = np.random.default_rng(seed)
            idx = local_rng.choice(n, self.psi, replace=False)
            return _build_tree(X_normal[idx], self.lmax, local_rng)

        seeds = rng.integers(0, 2 ** 31, size=self.t)
        self.forest = Parallel(n_jobs=-1)(
            delayed(_build_one)(int(s)) for s in seeds
        )

        if X_anomalies is not None and len(X_anomalies) > 0:
            for tree in self.forest:
                for x in X_anomalies:
                    _route_anomaly(x, tree)
            for tree in self.forest:
                _compute_Ca(tree)

        return self

    def _raw_scores(self, X):
        """Return mean path length h, and centroid distances dx, da per sample.

        h is averaged over all trees. dx (distance to the normal centroid) and
        da (distance to the anomaly centroid) are averaged only over the trees
        whose leaf actually has that centroid, matching Marteau's HIF; this
        avoids biasing da toward zero for leaves with no routed anomalies.
        """
        results = Parallel(n_jobs=-1)(
            delayed(_score_one_tree)(tree, X) for tree in self.forest
        )
        h = np.mean([r[0] for r in results], axis=0)
        dx_sum = np.sum([r[1] for r in results], axis=0)
        cx_cnt = np.sum([r[2] for r in results], axis=0)
        da_sum = np.sum([r[3] for r in results], axis=0)
        ca_cnt = np.sum([r[4] for r in results], axis=0)
        dx = np.where(cx_cnt > 0, dx_sum / np.maximum(cx_cnt, 1), 0.0)
        da = np.where(ca_cnt > 0, da_sum / np.maximum(ca_cnt, 1), 0.0)
        return h, dx, da

    def _signals(self, X):
        """Return the three raw signals (iso, dx, ratio) for the samples in X.

        iso = 2^(-h/c) is the isolation score (high for anomalies, short paths),
        matching Marteau's original HIF; dx is the distance to the normal-data
        centroid; ratio is dx divided by the distance to the anomaly centroid.
        All three increase with how anomalous a sample is.
        """
        h, dx, da = self._raw_scores(X)
        iso = 2.0 ** (-h / self._c) if self._c > 0 else np.zeros_like(h)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(da != 0, dx / da, 0.0)
        return iso, dx, ratio

    def calibrate(self, X_ref):
        """Fit and store the normalization bounds on a reference set.

        Called on the validation set before threshold selection so that the
        same min/max scaling is applied to the test set at inference time.
        """
        iso, dx, ratio = self._signals(X_ref)
        self._norm = (
            float(iso.min()), float(iso.max()),
            float(dx.min()), float(dx.max()),
            float(ratio.min()), float(ratio.max()),
        )
        return self

    @staticmethod
    def _scale(arr, lo, hi):
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    def score_samples(self, X, alpha1=None, alpha2=None):
        """Return anomaly scores in [0, 1]; higher means more anomalous.

        If calibrate() has been called, the stored normalization bounds are
        used; otherwise the bounds are taken from this batch.
        """
        a1 = alpha1 if alpha1 is not None else self.alpha1
        a2 = alpha2 if alpha2 is not None else self.alpha2

        iso, dx, ratio = self._signals(X)
        if self._norm is not None:
            iso_lo, iso_hi, sc_lo, sc_hi, sa_lo, sa_hi = self._norm
        else:
            iso_lo, iso_hi = iso.min(), iso.max()
            sc_lo, sc_hi = dx.min(), dx.max()
            sa_lo, sa_hi = ratio.min(), ratio.max()

        s = self._scale(iso, iso_lo, iso_hi)
        sc = self._scale(dx, sc_lo, sc_hi)
        sa = self._scale(ratio, sa_lo, sa_hi)

        return a2 * (a1 * s + (1.0 - a1) * sc) + (1.0 - a2) * sa

    def predict(self, X, threshold):
        return (self.score_samples(X) > threshold).astype(int)


def fbeta_threshold(scores, y_true, precision_weight, n_grid=200):
    """Select the threshold that maximises a precision-weighted F-beta score.

    beta = sqrt((1 - w) / w). With w = 0.9 this puts most of the weight on
    precision, matching the operating point reported in the paper.

    The search is a grid over the range of validation scores (Algorithm 1 in
    the paper). A grid is used rather than a continuous optimiser because the
    F-beta objective is piecewise constant in the threshold, which traps
    continuous solvers in flat regions.
    """
    beta = np.sqrt((1 - precision_weight) / precision_weight)
    b2 = beta ** 2

    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return lo

    grid = np.linspace(lo, hi, n_grid)
    best_thr, best_f = lo, -1.0
    for thr in grid:
        y_pred = (scores > thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        denom = b2 * p + r
        f = (1 + b2) * p * r / denom if denom > 0 else 0.0
        if f > best_f:
            best_f, best_thr = f, thr
    return float(best_thr)
