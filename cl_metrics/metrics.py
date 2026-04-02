"""
CLMetrics: Stateless computation of standard Continual Learning metrics
from a raw per-task accuracy matrix.

Implements the canonical formulations from:
- Diaz-Rodriguez et al. (2018): BWT, FWT, Intransigence
- Lopez-Paz & Ranzato (2017): AA
- Serra et al. (2018): Plasticity / Stability Index
"""

import numpy as np
from typing import Union, Optional


class CLMetrics:
    """
    Stateless CL metric engine. Accepts a raw N x N accuracy matrix
    and outputs all standard CIL evaluation metrics.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)
        R[i, j] = accuracy on task j after training on task i.
        Lower triangular = retention. Upper triangular = zero-shot transfer.
    task_weights : np.ndarray, optional
        Per-task class counts for micro-averaging. If None, macro-average.

    Examples
    --------
    >>> import numpy as np
    >>> from cl_metrics import CLMetrics
    >>> R = np.array([[0.90, 0.00, 0.00],
    ...               [0.72, 0.85, 0.00],
    ...               [0.65, 0.78, 0.88]])
    >>> m = CLMetrics(R)
    >>> m.summary()
    """

    def __init__(
        self,
        matrix: np.ndarray,
        task_weights: Optional[np.ndarray] = None
    ):
        matrix = np.array(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"matrix must be square (N x N). Got shape {matrix.shape}."
            )
        if np.any(matrix < 0) or np.any(matrix > 1):
            raise ValueError(
                "matrix values must be in [0, 1]. "
                "If passing percentages, divide by 100 first."
            )
        self.R = matrix
        self.N = matrix.shape[0]
        if task_weights is not None:
            task_weights = np.array(task_weights, dtype=float)
            if task_weights.shape != (self.N,):
                raise ValueError(
                    f"task_weights must have shape ({self.N},). "
                    f"Got {task_weights.shape}."
                )
            self.weights = task_weights / task_weights.sum()
        else:
            self.weights = np.ones(self.N) / self.N

    def average_accuracy(self) -> float:
        """
        Average Accuracy (AA): weighted mean of final accuracy across all tasks.

        AA = sum_j(w_j * R[N-1, j]) for j in 0..N-1

        Returns
        -------
        float
            Weighted average accuracy after all tasks are trained.
        """
        return float(np.dot(self.weights, self.R[-1, :]))

    def backward_transfer(self) -> float:
        """
        Backward Transfer (BWT): mean change in past-task accuracy
        after subsequent learning. Negative = forgetting.

        BWT = (1 / N-1) * sum_{j=0}^{N-2} (R[N-1, j] - R[j, j])

        Returns
        -------
        float
            BWT score. Negative values indicate catastrophic forgetting.
        """
        if self.N < 2:
            return 0.0
        diffs = [self.R[-1, j] - self.R[j, j] for j in range(self.N - 1)]
        return float(np.mean(diffs))

    def forward_transfer(self) -> float:
        """
        Forward Transfer (FWT): mean zero-shot performance on future tasks
        relative to a random baseline (assumed 0.0 by default).

        FWT = (1 / N-1) * sum_{j=1}^{N-1} (R[j-1, j] - b_j)
        where b_j = 0.0 (random chance baseline).

        Returns
        -------
        float
            FWT score. Positive = positive transfer.
        """
        if self.N < 2:
            return 0.0
        fwt_vals = [self.R[j - 1, j] for j in range(1, self.N)]
        return float(np.mean(fwt_vals))

    def intransigence(
        self,
        reference_accuracies: Optional[np.ndarray] = None
    ) -> float:
        """
        Intransigence: resistance to learning new tasks, measured as the
        gap between the continual learner and an isolated single-task oracle.

        I = (1 / N) * sum_{j=0}^{N-1} (ref_j - R[j, j])

        Follows strict Diaz-Rodriguez et al. (NeurIPS CL Workshop 2018)
        formulation. If reference_accuracies is None, uses R[j,j] as
        the self-reference (intransigence = 0.0 by definition).

        Parameters
        ----------
        reference_accuracies : np.ndarray, shape (N,), optional
            Oracle accuracy for each task when trained in isolation.

        Returns
        -------
        float
            Intransigence score. Higher = more rigid.
        """
        if reference_accuracies is None:
            return 0.0
        ref = np.array(reference_accuracies, dtype=float)
        if ref.shape != (self.N,):
            raise ValueError(
                f"reference_accuracies must have shape ({self.N},)."
            )
        gaps = [ref[j] - self.R[j, j] for j in range(self.N)]
        return float(np.mean(gaps))

    def plasticity_index(self) -> float:
        """
        Plasticity Index: mean diagonal accuracy (how well the model
        learns each new task at training time).

        PI = (1 / N) * sum_{j=0}^{N-1} R[j, j]

        Returns
        -------
        float
            Plasticity index in [0, 1]. Higher = more plastic.
        """
        return float(np.mean(np.diag(self.R)))

    def stability_index(self) -> float:
        """
        Stability Index: mean retention ratio across all tasks.
        For each task j, retention = R[N-1, j] / R[j, j] (clamped to [0,1]).

        SI = (1 / N) * sum_{j=0}^{N-1} (R[N-1, j] / max(R[j, j], eps))

        Returns
        -------
        float
            Stability index in [0, 1]. Higher = more stable.
        """
        eps = 1e-8
        ratios = [
            min(self.R[-1, j] / max(self.R[j, j], eps), 1.0)
            for j in range(self.N)
        ]
        return float(np.mean(ratios))

    def forgetting_measure(self) -> float:
        """
        Forgetting Measure: mean maximum drop in accuracy for each past task.

        F = (1 / N-1) * sum_{j=0}^{N-2} max_{l in j..N-1} (R[j,j] - R[l,j])

        Returns
        -------
        float
            Forgetting measure. Higher = more forgetting.
        """
        if self.N < 2:
            return 0.0
        forgetting = []
        for j in range(self.N - 1):
            max_drop = max(self.R[j, j] - self.R[l, j] for l in range(j, self.N))
            forgetting.append(max(max_drop, 0.0))
        return float(np.mean(forgetting))

    def summary(self) -> dict:
        """
        Compute and return all metrics as a dictionary.

        Returns
        -------
        dict
            Keys: AA, BWT, FWT, Plasticity, Stability, Forgetting.
        """
        results = {
            "AA":          round(self.average_accuracy(), 4),
            "BWT":         round(self.backward_transfer(), 4),
            "FWT":         round(self.forward_transfer(), 4),
            "Plasticity":  round(self.plasticity_index(), 4),
            "Stability":   round(self.stability_index(), 4),
            "Forgetting":  round(self.forgetting_measure(), 4),
        }
        print("\n=== cl-metrics Summary ===")
        for k, v in results.items():
            print(f"  {k:<12}: {v:.4f}")
        print("==========================\n")
        return results
