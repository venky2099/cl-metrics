# cl-metrics 📐

> **The missing `scikit-learn.metrics` for Continual Learning.**  
> Feed it a matrix. Get your metrics. No framework. No boilerplate. No pain.

[![PyPI version](https://badge.fury.io/py/cl-metrics.svg)](https://badge.fury.io/py/cl-metrics)
[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen)](https://github.com/venky2099/cl-metrics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19388144.svg)](https://doi.org/10.5281/zenodo.19388144)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--3315--7907-green)](https://orcid.org/0000-0002-3315-7907)

---

## Documentation

- 📖 [Interactive FAQ](https://venky2099.github.io/cl-metrics/faq.html) — 40 questions answered, searchable by category

---

## The Problem Every CIL Researcher Knows

You trained your model. You have your accuracy matrix. Now you need AA, BWT, FWT, Intransigence.

So you open Avalanche. Or PyCIL. Or FACIL.

And you find out that **every framework buries its metrics inside its own training loop.** You can't just pass a numpy array. You have to simulate their data stream, wrap your model in their classes, and fight their abstractions — just to compute a mean and a difference.

So you write your own NumPy script. Again. Like everyone else.

**cl-metrics ends this.**

---

## Install

```bash
pip install cl-metrics
```

---

## 30-Second Quick Start

```python
import numpy as np
from cl_metrics import CLMetrics

# Your N x N accuracy matrix
# R[i, j] = accuracy on task j after training on task i
R = np.array([
    [0.90, 0.00, 0.00],
    [0.72, 0.85, 0.00],
    [0.65, 0.78, 0.88],
])

m = CLMetrics(R)
m.summary()
```

```
=== cl-metrics Summary ===
  AA          : 0.7700
  BWT         : -0.1350
  FWT         : 0.0000
  Plasticity  : 0.8767
  Stability   : 0.8217
  Forgetting  : 0.1350
==========================
```

That's it. No imports beyond numpy. No framework. No training loop.

---

## What You Get

### Standard CIL Metrics

All implemented to their **canonical formulations** — no improvisation, no drift.

| Metric | What it measures | Canonical Reference |
|--------|-----------------|-------------------|
| **AA** | Mean final accuracy across all tasks | Lopez-Paz & Ranzato (2017) |
| **BWT** | How much new learning hurts old tasks (forgetting) | Díaz-Rodríguez et al. (2018) |
| **FWT** | Zero-shot performance on future tasks | Díaz-Rodríguez et al. (2018) |
| **Intransigence** | Resistance to learning new tasks vs. oracle | Díaz-Rodríguez et al. (2018) |
| **Plasticity Index** | How well the model learns each new task | Serra et al. (2018) |
| **Stability Index** | How much past knowledge is retained | Serra et al. (2018) |
| **Forgetting Measure** | Maximum accuracy drop per task | Chaudhry et al. (2018) |

### SNN Energy-Aware Metrics ⚡ *(First standardised suite)*

If you work with **Spiking Neural Networks**, accuracy alone is not enough. A model that achieves 90% accuracy at 40% spike rate is not better than one at 88% accuracy at 5% spike rate. Until now, there was no standard way to measure this.

```python
from cl_metrics import SNNMetrics

spike_rates = np.array([0.12, 0.09, 0.11])  # mean firing rate per task
snn = SNNMetrics(R, spike_rates=spike_rates)
snn.summary()
```

```
=== cl-metrics Summary ===
  AA          : 0.7700
  BWT         : -0.1350
  ...
==========================
=== SNN Energy Metrics ===
  SRP         : 0.1067   ← Spike Rate Proxy (energy proxy)
  SR-AA       : 0.6878   ← Accuracy penalised by energy cost
  EA-BWT      : -0.1208  ← Energy-weighted forgetting
  EER         : 2.1563   ← Error-to-Energy Ratio (lower = better)
==========================
```

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| **SRP** | mean(spike_rates) | Dynamic energy proxy |
| **SR-AA** | AA × (1 − SRP) | Accuracy adjusted for energy cost |
| **EA-BWT** | Energy-weighted BWT per task | High-energy forgetting penalised more |
| **EER** | (1 − AA) / SRP | Combined error + energy in one scalar |

---

## Full API Reference

### `CLMetrics(matrix, task_weights=None)`

```python
from cl_metrics import CLMetrics

m = CLMetrics(R)                          # macro-average (equal task weights)
m = CLMetrics(R, task_weights=[10,20,10]) # micro-average by class count

m.average_accuracy()     # → float
m.backward_transfer()    # → float (negative = forgetting)
m.forward_transfer()     # → float
m.intransigence(ref)     # → float (pass oracle accuracies per task)
m.plasticity_index()     # → float
m.stability_index()      # → float
m.forgetting_measure()   # → float
m.summary()              # → dict (prints + returns all metrics)
```

### `SNNMetrics(matrix, spike_rates=None, task_weights=None)`

```python
from cl_metrics import SNNMetrics

snn = SNNMetrics(R, spike_rates=np.array([0.12, 0.09, 0.11]))

snn.spike_rate_proxy()           # → float
snn.spike_rate_normalized_aa()   # → float
snn.energy_adjusted_bwt()        # → float
snn.energy_to_error_ratio()      # → float
snn.summary()                    # → dict (all CL + SNN metrics)
```

### Input Format

```
R[i, j] = accuracy on task j, evaluated after training on task i

         Task 0   Task 1   Task 2
After 0 [ 0.90    0.00     0.00  ]   ← only trained on task 0
After 1 [ 0.72    0.85     0.00  ]   ← trained on tasks 0-1
After 2 [ 0.65    0.78     0.88  ]   ← trained on all tasks

- Values must be in [0, 1]  (not percentages)
- Shape must be (N, N)
- Lower triangle = retention | Upper triangle = zero-shot transfer
```

---

## Why Not Just Use Avalanche / PyCIL?

| | Avalanche / PyCIL | **cl-metrics** |
|---|---|---|
| Input | Requires live data stream + model | Raw numpy array |
| Framework dependency | PyTorch required | numpy only |
| Works with JAX / TF / C++ | ❌ | ✅ |
| Works with neuromorphic chips | ❌ | ✅ |
| SNN energy metrics | ❌ | ✅ |
| Lines of code to get BWT | ~50 (wrapper code) | 3 |
| Install size | Heavy | ~2MB |

---

## Intransigence: Pass Your Oracle

```python
# oracle_accs[j] = accuracy of a model trained *only* on task j
oracle_accs = np.array([0.92, 0.89, 0.91])
m.intransigence(reference_accuracies=oracle_accs)
```

If you don't have oracle accuracies, intransigence returns 0.0 by default
(mathematically correct — the model is its own reference).

---

## Validated Against

cl-metrics metrics are validated against the **Maya Research Series** (P3–P7),
a 7-paper neuromorphic SNN continual learning benchmark on Split-CIFAR-10 and Split-CIFAR-100.

| Paper | Benchmark | Reported AA | Reported BWT |
|-------|-----------|-------------|--------------|
| Maya-CL (P3) | Split-CIFAR-10 TIL | 62.38% | — |
| Maya-Smriti (P4) | Split-CIFAR-10 CIL | 31.84% | — |
| Maya-Viveka (P5) | Split-CIFAR-100 CIL | 16.03% | −50.50% |
| Maya-Chitta (P6) | Split-CIFAR-100 CIL | 14.42% | — |
| Maya-Manas (P7) | Split-CIFAR-100 CIL | 15.19% | −50.91% |

DOIs: [P3](https://doi.org/10.5281/zenodo.19201769) · [P4](https://doi.org/10.5281/zenodo.19228975) · [P5](https://doi.org/10.5281/zenodo.19279002) · [P6](https://doi.org/10.5281/zenodo.19337041) · [P7](https://doi.org/10.5281/zenodo.19363006)

---

## The Reproducibility Problem This Fixes

The CIL community has a well-documented metric inconsistency crisis:

- **AA** is computed with both macro-averaging (by task) and micro-averaging (by class count) — these give different numbers
- **BWT** formulations differ in how they handle early stopping and buffer sizes
- **FWT** has two completely different definitions in common use (zero-shot vs. curriculum acceleration)
- **Intransigence** is routinely approximated without the oracle, breaking comparability

`cl-metrics` implements each metric to its **original published formulation**, documented and unit-tested. When you report metrics computed with `cl-metrics`, reviewers can verify your numbers independently.

---

## Contributing

Found a metric formulation that differs from what's implemented? Open an issue with the paper reference. Correctness over convenience — always.

```bash
git clone https://github.com/venky2099/cl-metrics
cd cl-metrics
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Citation

If `cl-metrics` saved you from writing another NumPy script, please cite:

```bibtex
@software{swaminathan2026clmetrics,
  author       = {Swaminathan, Venkatesh},
  title        = {cl-metrics: Stateless Continual Learning Evaluation Metrics
                  with SNN Energy-Aware Extensions},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19388144},
  url          = {https://doi.org/10.5281/zenodo.19388144},
  orcid        = {0000-0002-3315-7907}
}```

---

## Author

**Venkatesh Swaminathan**  
Founder, Nexus Learning Labs · Bengaluru, India  
M.Sc. Data Science & AI, BITS Pilani  
ORCID: [0000-0002-3315-7907](https://orcid.org/0000-0002-3315-7907)  
GitHub: [@venky2099](https://github.com/venky2099)

---

*Built because the community deserved a tool that just works.*
