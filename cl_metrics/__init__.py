"""
cl-metrics: A stateless, architecture-agnostic Python library for
Continual Learning and Class-Incremental Learning evaluation metrics.

Author: Venkatesh Swaminathan
Affiliation: Nexus Learning Labs, Bengaluru | M.Sc. DS&AI, BITS Pilani
ORCID: 0000-0002-3315-7907
GitHub: venky2099
"""

from .metrics import CLMetrics
from .snn_metrics import SNNMetrics

__version__ = "0.1.0"
__author__ = "Venkatesh Swaminathan"
__all__ = ["CLMetrics", "SNNMetrics"]
