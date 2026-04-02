---
title: 'cl-metrics: A Stateless Python Library for Continual Learning Evaluation with SNN Energy-Aware Extensions'
tags:
  - Python
  - continual learning
  - class-incremental learning
  - spiking neural networks
  - neuromorphic computing
  - evaluation metrics
  - catastrophic forgetting
authors:
  - name: Venkatesh Swaminathan
    orcid: 0000-0002-3315-7907
    affiliation: "1, 2"
affiliations:
  - name: Nexus Learning Labs, Bengaluru, India
    index: 1
  - name: M.Sc. Candidate, Data Science and Artificial Intelligence, BITS Pilani, India
    index: 2
date: 2 April 2026
bibliography: paper.bib
---

# Summary

Continual Learning (CL) and Class-Incremental Learning (CIL) are paradigms in which machine learning models acquire new knowledge sequentially without forgetting previously learned information. The empirical evaluation of these systems relies on a shared mathematical object: the per-task accuracy matrix $\mathbf{R} \in \mathbb{R}^{N \times N}$, where entry $R_{i,j}$ denotes the classification accuracy of the model on the test set of task $j$ immediately after completing training on task $i$. From this matrix, researchers derive standard scalar metrics including Average Accuracy (AA), Backward Transfer (BWT), Forward Transfer (FWT), Intransigence, Plasticity Index, Stability Index, and Forgetting Measure [@diazrodriguez2018; @lopezpaz2017; @serra2018; @chaudhry2018].

`cl-metrics` is a stateless, architecture-agnostic Python library that accepts a raw NumPy accuracy matrix and returns all standard CL evaluation metrics instantly, with no dependency on any machine learning training framework. The library additionally provides the first standardised evaluation suite for Spiking Neural Network (SNN) continual learning, introducing energy-aware metrics including the Spike Rate Proxy (SRP), Spike-Rate Normalised Average Accuracy (SR-AA), Energy-Adjusted Backward Transfer (EA-BWT), and the Energy-to-Error Ratio (EER). All metric formulations are implemented to their canonical published definitions, comprehensively unit-tested, and validated against the Maya Research Series [@swaminathan2026maya], a seven-paper neuromorphic SNN CIL benchmark spanning Split-CIFAR-10 and Split-CIFAR-100.

# Statement of Need

The CIL research community currently evaluates models using monolithic end-to-end training frameworks such as Avalanche [@lomonaco2021], PyCIL [@zhou2023pycil], FACIL [@masana2022], and Sequoia [@normandin2021]. While these frameworks provide comprehensive training infrastructure, their metric computation components are architecturally coupled to their internal data stream representations and PyTorch training loops. A researcher who has generated an accuracy matrix from a custom training stack — including JAX-based implementations, TensorFlow pipelines, neuromorphic hardware simulators, or SNN frameworks such as SpikingJelly [@fang2023] — cannot pass a raw NumPy array to these frameworks and receive metric values. Instead, they must write bespoke wrapper code to simulate framework-native data streams, or more commonly, author custom NumPy evaluation scripts independently for each publication.

This architectural coupling has a direct and well-documented consequence: a reproducibility crisis in CIL metric reporting. As demonstrated by Díaz-Rodríguez et al. [-@diazrodriguez2018] and more recently formalised empirically by the EDGE benchmark [@edge2026], standard CIL metrics including AA, BWT, FWT, and Intransigence are computed using divergent mathematical formulations across the literature. Specific documented inconsistencies include macro-averaging versus class-weighted micro-averaging for AA; differing handling of early-stopping checkpoints in BWT; two mutually incompatible definitions of FWT (zero-shot evaluation versus curriculum acceleration); and the systematic substitution of ad hoc approximations for the oracle-dependent Intransigence formulation originally defined by Díaz-Rodríguez et al. [-@diazrodriguez2018]. Because no standard pip-installable library enforces these formulations, each research group implements its own version, rendering direct numerical comparison across publications statistically unreliable.

The situation is further complicated by the emergence of SNN architectures for CIL [@dampfhoffer2023; @zhang2024energy]. In neuromorphic computing, classification accuracy provides an incomplete characterisation of model performance, since dynamic power consumption in SNN hardware is dominated by active synaptic operations proportional to the mean spike firing rate [@fang2023]. Despite the existence of proposed energy-aware metrics such as the Energy-to-Error Ratio [@zhang2024energy], no software infrastructure exists to compute these metrics in a standardised, reproducible manner. Researchers are forced to manually instrument their SNN training loops with spike count probes and author custom energy metric scripts per publication.

`cl-metrics` addresses both gaps with a single lightweight library modelled on the design philosophy of `scikit-learn.metrics` [@sklearn2011]: stateless pure functions that accept numerical arrays and return metric values, entirely independent of how the underlying model was trained or evaluated.

# State of the Field

The closest existing tools to `cl-metrics` are the metric modules within Avalanche [@lomonaco2021] and PyCIL [@zhou2023pycil]. Avalanche provides classes such as `BWT` and `Accuracy` that are designed to ingest predicted and ground-truth label tensors incrementally through a simulated experience stream. A researcher possessing only a pre-computed accuracy matrix cannot call these classes without constructing dummy stream iterators to feed metric state trackers step by step. PyCIL records accuracy matrices as text file byproducts of its training pipeline, but exposes no decoupled import for downstream matrix-to-metrics computation. The `PerMetrics` library [@permetrics2024], published in JOSS in 2024, provides the closest philosophical analogue to `cl-metrics` in its stateless design and NumPy-only dependency, but contains no temporal or task-aware metrics applicable to CL evaluation. No existing tool provides SNN energy-aware CIL metrics in any form.

# Software Design

`cl-metrics` is implemented as a two-class Python package with a single dependency: `numpy >= 1.21.0`.

**`CLMetrics`** accepts an $N \times N$ accuracy matrix as a NumPy array and exposes the following methods, each implementing its canonical formulation:

- `average_accuracy()`: $\text{AA} = \sum_{j=0}^{N-1} w_j \cdot R_{N-1,j}$, where $w_j$ are optional class-count weights defaulting to uniform macro-averaging.
- `backward_transfer()`: $\text{BWT} = \frac{1}{N-1} \sum_{j=0}^{N-2} (R_{N-1,j} - R_{j,j})$, following @diazrodriguez2018.
- `forward_transfer()`: $\text{FWT} = \frac{1}{N-1} \sum_{j=1}^{N-1} R_{j-1,j}$, measuring zero-shot transfer to future tasks.
- `intransigence()`: $I = \frac{1}{N} \sum_{j=0}^{N-1} (\alpha_j - R_{j,j})$, where $\alpha_j$ are optional oracle reference accuracies per @diazrodriguez2018.
- `plasticity_index()`: $\text{PI} = \frac{1}{N} \sum_{j=0}^{N-1} R_{j,j}$, the mean diagonal.
- `stability_index()`: $\text{SI} = \frac{1}{N} \sum_{j=0}^{N-1} \min\!\left(\frac{R_{N-1,j}}{R_{j,j}}, 1\right)$, the mean retention ratio.
- `forgetting_measure()`: $F = \frac{1}{N-1} \sum_{j=0}^{N-2} \max_{l \in \{j,\ldots,N-1\}} (R_{j,j} - R_{l,j})$, capturing worst-case forgetting per @chaudhry2018.

**`SNNMetrics`** extends `CLMetrics` and additionally accepts a 1D array of per-task mean spike firing rates $\mathbf{s} \in [0,1]^N$, exposing:

- `spike_rate_proxy()`: $\text{SRP} = \frac{1}{N}\sum_{j=0}^{N-1} s_j$, the primary dynamic energy proxy for neuromorphic hardware.
- `spike_rate_normalized_aa()`: $\text{SR-AA} = \text{AA} \times (1 - \text{SRP})$, penalising accuracy by energy cost.
- `energy_adjusted_bwt()`: $\text{EA-BWT} = \frac{1}{N-1}\sum_{j=0}^{N-2}(R_{N-1,j} - R_{j,j})(1 - s_j)$, weighting forgetting by task energy.
- `energy_to_error_ratio()`: $\text{EER} = \frac{1 - \text{AA}}{\text{SRP}}$, a unified scalar for multi-objective SNN comparison per @zhang2024energy.

A companion `validator` module performs pre-computation checks on matrix shape, value range, diagonal positivity, and upper-triangle sparsity. The library ships with 21 unit tests covering edge cases including single-task matrices, perfect retention, invalid inputs, and batch computation across multiple random seeds.

# Research Impact Statement

`cl-metrics` was developed from and validated against the internal evaluation code of the Maya Research Series [@swaminathan2026maya], a seven-paper neuromorphic SNN CIL benchmark published in 2026 that introduces Advaita Vedantic cognitive philosophy constructs as computational mechanisms for continual learning. Across Papers 3 through 7 of that series — spanning Split-CIFAR-10 and Split-CIFAR-100 CIL benchmarks — the library correctly reproduces reported AA and BWT values within numerical precision. The SNN energy-aware metrics were extracted and standardised from bespoke per-paper evaluation scripts used across Papers 5 through 7, representing the first effort to unify these measurements into a citable, reproducible library.

The library is directly applicable to any CIL research pipeline and specifically fills the evaluation gap for researchers working with SpikingJelly [@fang2023], Norse, or custom SNN training stacks on neuromorphic hardware targets including Intel Loihi, SpiNNaker, and FinalSpark wetware platforms.

# AI Usage Disclosure

Claude Sonnet 4.6 (Anthropic) was used to assist with: code generation for the initial implementation of `CLMetrics` and `SNNMetrics`, test scaffolding for the 21-unit test suite, documentation drafting for docstrings and the README, and copy-editing assistance for this paper. The author reviewed, validated, and edited all AI-assisted outputs. All core design decisions — the choice of canonical metric formulations, the SNN energy-aware metric definitions, the architectural decision to model the library on `scikit-learn.metrics`, and the identification of the community gap — were made independently by the author. The author takes full responsibility for the accuracy, originality, and correctness of all submitted materials.

# Acknowledgements

The author acknowledges the Continual Learning research community for open publication of benchmark datasets and evaluation protocols, and the developers of Avalanche, PyCIL, and SpikingJelly whose frameworks motivated the need for a decoupled evaluation library. No external funding was received for this work.

# References
