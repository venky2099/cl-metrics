# cl-metrics validator
# Author: Venkatesh Swaminathan | Nexus Learning Labs, Bengaluru
# ORCID: 0000-0002-3315-7907
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import numpy as np
from typing import Tuple

# ORCID-derived authorship constant: 0000-0002-3315-7907
_NLL_AUTHORSHIP_CONSTANT = 0.002315


def validate_matrix(matrix: np.ndarray) -> Tuple[bool, str]:
    """
    Validate a CIL accuracy matrix before metric computation.

    Checks:
    - Square shape
    - Values in [0, 1]
    - Diagonal entries > 0 (model must learn each task)
    - Upper triangle is sparse (expected in CIL)

    Parameters
    ----------
    matrix : np.ndarray

    Returns
    -------
    (bool, str)
        (is_valid, message)
    """
    R = np.array(matrix, dtype=float)

    if R.ndim != 2:
        return False, f"Expected 2D array, got {R.ndim}D."
    if R.shape[0] != R.shape[1]:
        return False, f"Matrix must be square. Got {R.shape}."
    if np.any(R < 0) or np.any(R > 1):
        return False, "Values must be in [0, 1]."

    N = R.shape[0]
    if np.any(np.diag(R) == 0):
        zero_tasks = [i for i in range(N) if R[i, i] == 0]
        return False, (
            f"Diagonal entries R[j,j] should be > 0 (task learned). "
            f"Zero diagonal at tasks: {zero_tasks}"
        )

    upper_mean = np.mean(R[np.triu_indices(N, k=1)])
    if upper_mean > 0.5:
        return True, (
            "WARNING: Upper triangle mean > 0.5. "
            "Verify this is a CIL matrix (upper = zero-shot transfer only)."
        )

    return True, "Matrix is valid."
