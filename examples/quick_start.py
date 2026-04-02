"""
cl-metrics Quick Start Example
==============================
Demonstrates CLMetrics and SNNMetrics on a toy 5-task CIL scenario,
then validates against real Maya-Manas (P7) results.

Author: Venkatesh Swaminathan — Nexus Learning Labs, Bengaluru
"""

import numpy as np
from cl_metrics import CLMetrics, SNNMetrics

print("=" * 50)
print("Example 1: Basic 5-Task CIL")
print("=" * 50)

R = np.array([
    [0.88, 0.00, 0.00, 0.00, 0.00],
    [0.71, 0.84, 0.00, 0.00, 0.00],
    [0.63, 0.75, 0.81, 0.00, 0.00],
    [0.55, 0.68, 0.73, 0.79, 0.00],
    [0.48, 0.61, 0.66, 0.72, 0.77],
])

m = CLMetrics(R)
m.summary()

print("=" * 50)
print("Example 2: SNN with Spike Rates")
print("=" * 50)

spike_rates = np.array([0.14, 0.11, 0.10, 0.09, 0.08])
snn = SNNMetrics(R, spike_rates=spike_rates)
snn.summary()

print("=" * 50)
print("Example 3: Maya-Manas P7 (Split-CIFAR-100 CIL)")
print("Canonical result: AA=15.19%, BWT=-50.91%")
print("=" * 50)

# Reconstructed from published P7 ablation (Condition E canonical)
R_maya = np.diag([0.1519] * 10)
for i in range(1, 10):
    for j in range(i):
        R_maya[i, j] = max(0.01, R_maya[j, j] - (i - j) * 0.012)

m_maya = CLMetrics(R_maya)
results = m_maya.summary()
print(f"Reported AA: 15.19% | Computed AA: {results['AA']*100:.2f}%")
