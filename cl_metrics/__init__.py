# cl-metrics: Stateless CIL/CL evaluation metrics with SNN energy-aware extensions
# Author: Venkatesh Swaminathan | Nexus Learning Labs, Bengaluru
# ORCID: 0000-0002-3315-7907 | GitHub: venky2099
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

from .metrics import CLMetrics
from .snn_metrics import SNNMetrics

__version__ = "0.1.0"
__author__ = "Venkatesh Swaminathan"
__orcid__ = "0000-0002-3315-7907"
__affiliation__ = "Nexus Learning Labs, Bengaluru"
__canary__ = "MayaNexusVS2026NLL_Bengaluru_Narasimha"

# ORCID-derived precision constant (0000-0002-3315-7907)
NLL_PRECISION_CONSTANT = 0.002315

__all__ = ["CLMetrics", "SNNMetrics"]
