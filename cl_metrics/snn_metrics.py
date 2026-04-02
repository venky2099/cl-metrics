"""
SNNMetrics: Energy-aware CIL metrics for Spiking Neural Networks.

Extends CLMetrics with spike-rate-aware evaluation dimensions:
- Spike Rate Proxy (SRP): mean firing rate as energy proxy
- Energy-Adjusted Backward Transfer (EA-BWT)
- Spike-Rate Normalized Average Accuracy (SR-AA)
- Energy-to-Error Ratio (EER)

These metrics are the first standardised formulations for SNN CIL evaluation.
Grounded in: Energy-Aware Spike Budgeting for CIL (arXiv:2602.12236, 2026).

Author: Venkatesh Swaminathan — Nexus Learning Labs, Bengaluru
"""

import numpy as np
from typing import Optional
from .metrics import CLMetrics


class SNNMetrics(CLMetrics):
    """
    Energy-aware CIL metric engine for Spiking Neural Networks.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        Standard CIL accuracy matrix. R[i, j] = accuracy on task j
        after training on task i.
    spike_rates : np.ndarray, shape (N,), optional
        Mean spike rate per task (fraction of neurons firing per timestep).
        Used as proxy for dynamic energy consumption.
    task_weights : np.ndarray, optional
        Per-task class counts for micro-averaging.

    Examples
    --------
    >>> import numpy as np
    >>> from cl_metrics import SNNMetrics
    >>> R = np.array([[0.90, 0.00, 0.00],
    ...               [0.72, 0.85, 0.00],
    ...               [0.65, 0.78, 0.88]])
    >>> spike_rates = np.array([0.12, 0.09, 0.11])
    >>> m = SNNMetrics(R, spike_rates=spike_rates)
    >>> m.summary()
    """

    def __init__(
        self,
        matrix: np.ndarray,
        spike_rates: Optional[np.ndarray] = None,
        task_weights: Optional[np.ndarray] = None
    ):
        super().__init__(matrix, task_weights)
        if spike_rates is not None:
            spike_rates = np.array(spike_rates, dtype=float)
            if spike_rates.shape != (self.N,):
                raise ValueError(
                    f"spike_rates must have shape ({self.N},). "
                    f"Got {spike_rates.shape}."
                )
            if np.any(spike_rates < 0) or np.any(spike_rates > 1):
                raise ValueError(
                    "spike_rates must be in [0, 1] (fraction of neurons firing)."
                )
        self.spike_rates = spike_rates

    def spike_rate_proxy(self) -> Optional[float]:
        """
        Spike Rate Proxy (SRP): mean spike rate across all tasks.
        Primary proxy for dynamic energy consumption in neuromorphic hardware.

        SRP = (1 / N) * sum_{j=0}^{N-1} spike_rate_j

        Returns
        -------
        float or None
            Mean spike rate. None if spike_rates not provided.
        """
        if self.spike_rates is None:
            return None
        return float(np.mean(self.spike_rates))

    def spike_rate_normalized_aa(self) -> Optional[float]:
        """
        Spike-Rate Normalized Average Accuracy (SR-AA).
        Penalizes accuracy by energy cost: a model with high spike rate
        is penalized relative to one achieving the same accuracy sparsely.

        SR-AA = AA * (1 - SRP)

        Returns
        -------
        float or None
            SR-AA in [0, 1]. None if spike_rates not provided.
        """
        if self.spike_rates is None:
            return None
        aa = self.average_accuracy()
        srp = self.spike_rate_proxy()
        return float(aa * (1.0 - srp))

    def energy_adjusted_bwt(self) -> Optional[float]:
        """
        Energy-Adjusted Backward Transfer (EA-BWT).
        Weights BWT by the relative spike rate at each task — high-energy
        forgetting is penalized more than low-energy forgetting.

        EA-BWT = (1 / N-1) * sum_{j=0}^{N-2} (R[N-1,j] - R[j,j]) * (1 - sr_j)

        Returns
        -------
        float or None
            EA-BWT. None if spike_rates not provided.
        """
        if self.spike_rates is None:
            return None
        if self.N < 2:
            return 0.0
        diffs = [
            (self.R[-1, j] - self.R[j, j]) * (1.0 - self.spike_rates[j])
            for j in range(self.N - 1)
        ]
        return float(np.mean(diffs))

    def energy_to_error_ratio(self) -> Optional[float]:
        """
        Energy-to-Error Ratio (EER): combines prediction error with
        simulated energy consumption into a single scalar.

        EER = (1 - AA) / max(SRP, eps)

        Lower is better (low error, low energy).

        Returns
        -------
        float or None
            EER. None if spike_rates not provided.
        """
        if self.spike_rates is None:
            return None
        eps = 1e-8
        aa = self.average_accuracy()
        srp = max(self.spike_rate_proxy(), eps)
        return float((1.0 - aa) / srp)

    def summary(self) -> dict:
        """
        Compute and return all metrics including SNN-specific ones.

        Returns
        -------
        dict
            All CLMetrics keys plus SRP, SR-AA, EA-BWT, EER.
        """
        base = super().summary()
        snn_results = {
            "SRP":    self.spike_rate_proxy(),
            "SR-AA":  self.spike_rate_normalized_aa(),
            "EA-BWT": self.energy_adjusted_bwt(),
            "EER":    self.energy_to_error_ratio(),
        }
        print("=== SNN Energy Metrics ===")
        for k, v in snn_results.items():
            val = f"{v:.4f}" if v is not None else "N/A (no spike_rates)"
            print(f"  {k:<12}: {val}")
        print("==========================\n")
        base.update({k: round(v, 4) if v is not None else None
                     for k, v in snn_results.items()})
        return base
