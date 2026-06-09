"""Hybrid Isolation Forest for network anomaly detection on CICIDS2017."""

from .ensemble import HIFEnsemble
from .forest import HybridIsolationForest

__all__ = ["HybridIsolationForest", "HIFEnsemble"]
__version__ = "1.0.0"
