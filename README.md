# cl-metrics

**A stateless, architecture-agnostic Python library for Continual Learning (CL) and Class-Incremental Learning (CIL) evaluation metrics.**

[![PyPI version](https://badge.fury.io/py/cl-metrics.svg)](https://badge.fury.io/py/cl-metrics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--3315--7907-green)](https://orcid.org/0000-0002-3315-7907)

---

## The Problem

Every major CIL framework (Avalanche, PyCIL, FACIL) computes metrics **inside their training loops**. If you trained your model on custom hardware, an SNN chip, or a non-PyTorch stack — you cannot extract standard metrics without writing wrapper code.

There is no scikit-learn.metrics equivalent for Continual Learning. **Until now.**

---

## What cl-metrics does

Feed it a raw N×N accuracy matrix. Get all standard CIL metrics instantly.
`python
import numpy as np
from cl_metrics import CLMetrics

R = np.array([
    [0.90, 0.00, 0.00],
    [0.72, 0.85, 0.00],
    [0.65, 0.78, 0.88],
])

m = CLMetrics(R)
m.summary()
`
`
=== cl-metrics Summary ===
  AA          : 0.7700
  BWT         : -0.1350
  FWT         : 0.0000
  Plasticity  : 0.8767
  Stability   : 0.8217
  Forgetting  : 0.1350
==========================
`

---

## SNN Energy-Aware Metrics

The first standardised SNN-specific CIL evaluation suite:
`python
from cl_metrics import SNNMetrics

spike_rates = np.array([0.12, 0.09, 0.11])
snn = SNNMetrics(R, spike_rates=spike_rates)
snn.summary()
`

| Metric | Description |
|--------|-------------|
| SRP    | Spike Rate Proxy — mean firing rate as energy proxy |
| SR-AA  | Spike-Rate Normalized Average Accuracy |
| EA-BWT | Energy-Adjusted Backward Transfer |
| EER    | Energy-to-Error Ratio |

---

## Installation
`ash
pip install cl-metrics
`

Or from source:
`ash
git clone https://github.com/venky2099/cl-metrics
cd cl-metrics
pip install -e .
`

---

## Metrics Reference

| Metric | Formula | Reference |
|--------|---------|-----------|
| AA | Mean of final row | Lopez-Paz & Ranzato (2017) |
| BWT | Mean drop in past-task accuracy | Diaz-Rodriguez et al. (2018) |
| FWT | Mean zero-shot on future tasks | Diaz-Rodriguez et al. (2018) |
| Intransigence | Gap to oracle | Diaz-Rodriguez et al. (2018) |
| Plasticity | Mean diagonal | Serra et al. (2018) |
| Stability | Mean retention ratio | Serra et al. (2018) |
| Forgetting | Mean max drop | Chaudhry et al. (2018) |
| SRP | Mean spike rate | arXiv:2602.12236 (2026) |
| SR-AA | AA × (1 - SRP) | This work |
| EA-BWT | Energy-weighted BWT | This work |
| EER | Error / Energy | arXiv:2602.12236 (2026) |

---

## Validated Against

Maya Research Series (P3–P7) — Split-CIFAR-10 and Split-CIFAR-100 CIL benchmarks.
DOIs: [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) through [10.5281/zenodo.19363006](https://doi.org/10.5281/zenodo.19363006)

---

## Author

**Venkatesh Swaminathan**
Nexus Learning Labs, Bengaluru | M.Sc. DS&AI, BITS Pilani
ORCID: [0000-0002-3315-7907](https://orcid.org/0000-0002-3315-7907)
GitHub: [venky2099](https://github.com/venky2099)

---

## Citation

If you use cl-metrics in your research, please cite:
`ibtex
@software{swaminathan2026clmetrics,
  author = {Swaminathan, Venkatesh},
  title  = {cl-metrics: Stateless CIL Evaluation Metrics with SNN Energy-Aware Extensions},
  year   = {2026},
  url    = {https://github.com/venky2099/cl-metrics},
  orcid  = {0000-0002-3315-7907}
}
`

---

## License

MIT License. See [LICENSE](LICENSE) for details.
