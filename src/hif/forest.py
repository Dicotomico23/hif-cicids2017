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
from scipy.optimize import minimize_scalar
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
    """Traverse one tree and return (h_x, delta_x, delta_a_x)."""
    e = 0
    node = tree
    while not isinstance(node, exNode):
        if x[node.splitDim] < node.splitVal:
            node = node.left
        else:
            node = node.right
        e += 1

    h_x = e + node.c_lenS
    delta_x = float(np.linalg.norm(x - node.CS)) if node.CS is not None else 0.0
    delta_ax = float(np.linalg.norm(x - node.Ca)) if node.Ca is not None else 0.0
    return h_x, delta_x, delta_ax


def _score_one_tree(tree, X):
    """Score every sample against a single tree (used by joblib workers)."""
    n = len(X)
    h = np.zeros(n)
    dx = np.zeros(n)
    da = np.zeros(n)
    for i in range(n):
        h[i], dx[i], da[i] = _score_sample(X[i], tree)
    return h, dx, da


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
        """Return the mean h, delta_x and delta_ax across all trees."""
        results = Parallel(n_jobs=-1)(
            delayed(_score_one_tree)(tree, X) for tree in self.forest
        )
        h = np.mean([r[0] for r in results], axis=0)
        dx = np.mean([r[1] for r in results], axis=0)
        da = np.mean([r[2] for r in results], axis=0)
        return h, dx, da

    @staticmethod
    def _normalize(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    def score_samples(self, X, alpha1=None, alpha2=None):
        """Return anomaly scores in [0, 1]; higher means more anomalous."""
        a1 = alpha1 if alpha1 is not None else self.alpha1
        a2 = alpha2 if alpha2 is not None else self.alpha2

        h, dx, da = self._raw_scores(X)
        s = self._normalize(h)
        sc = self._normalize(dx)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(da != 0, dx / da, 0.0)
        sa = self._normalize(ratio)

        return a2 * (a1 * s + (1.0 - a1) * sc) + (1.0 - a2) * sa

    def predict(self, X, threshold):
        return (self.score_samples(X) > threshold).astype(int)


def fbeta_threshold(scores, y_true, precision_weight):
    """Select the threshold that maximises a precision-weighted F-beta score.

    beta = sqrt((1 - w) / w). With w = 0.9 this puts most of the weight on
    precision, matching the operating point reported in the paper.
    """
    beta = np.sqrt((1 - precision_weight) / precision_weight)

    def neg_fbeta(thr):
        y_pred = (scores > thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        denom = beta ** 2 * p + r
        return -(1 + beta ** 2) * p * r / denom if denom > 0 else 0.0

    result = minimize_scalar(
        neg_fbeta,
        bounds=(float(scores.min()), float(scores.max())),
        method="bounded",
    )
    return float(result.x)
