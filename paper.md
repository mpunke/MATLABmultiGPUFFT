````markdown
---
title: "Multi-GPU Fast Fourier Transforms in MATLAB for Large-Scale Phase-Field Crystal Simulations"
tags:
  - MATLAB
  - fast Fourier transform
  - GPU computing
  - phase-field crystal
  - pseudo-spectral method
authors:
  - name: Maik Punke
    affiliation: 1
  - name: Marco Salvalaglio
    affiliation: 1
affiliations:
  - name: Institute of Scientific Computing, TU Dresden, Dresden, Germany
    index: 1
date: 2026
bibliography:
---

# Summary

We present a MATLAB-based framework for two- and three-dimensional fast Fourier transforms (FFTs) on multiple GPUs for large-scale numerical simulations using the pseudo-spectral Fourier method. The software implements two complementary multi-GPU strategies that overcome single-GPU memory limitations and accelerate spectral solvers.

The framework is motivated by phase-field crystal (PFC) models, which are governed by high-order partial differential equations and require fine spatial resolution in periodic domains. The implementation enables large-scale simulations and achieves significant performance improvements compared to CPU-based approaches.


