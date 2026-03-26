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
bibliography: paper.bib
---

# Summary

We present a MATLAB-based framework for two- and three-dimensional fast Fourier transforms (FFTs) on multiple GPUs for large-scale numerical simulations using the pseudo-spectral Fourier method. The software implements two complementary multi-GPU strategies that overcome single-GPU memory limitations and accelerate spectral solvers.

The framework is motivated by phase-field crystal (PFC) models, which are governed by high-order partial differential equations and require fine spatial resolution in periodic domains. The implementation enables large-scale simulations and achieves significant performance improvements compared to CPU-based approaches.

# Statement of need

The pseudo-spectral Fourier method [@boyd2001chebyshev] is widely used for solving partial differential equations with periodic boundary conditions due to its high accuracy and efficiency. However, its performance is dominated by repeated multidimensional FFTs, which become a bottleneck for large-scale simulations.

Phase-field crystal (PFC) models [@Elder2002; @Elder2004; @Emmerich2012] require large computational domains and fine spatial resolution, resulting in substantial memory demands. These limitations are particularly restrictive for single-GPU implementations and even more pronounced in multiphysics extensions involving multiple coupled fields [@skogvoll2022hydrodynamic; @Punke_2022].

This work addresses this limitation by introducing multi-GPU FFT strategies implemented in MATLAB, enabling scalable pseudo-spectral simulations and extending MATLAB's capabilities to distributed GPU computing.

# State of the field

Existing approaches to FFT-based pseudo-spectral solvers for PFC models include CPU-based parallel implementations [@pinomaa2024openpfc; @skogvoll2024comfit] and single-GPU solutions [@hallberg2025pypfc]. Distributed multi-GPU FFT libraries such as P3DFFT [@pekurovsky2012p3dFFT] and heFFTe [@ayala2020heFFTe] provide scalable FFT implementations but are typically implemented in low-level languages and are not directly accessible within MATLAB.

This creates a gap for MATLAB users who require high-performance spectral solvers with multi-GPU capabilities. The present work fills this gap by providing a MATLAB-native implementation of distributed FFT strategies.

# Mathematical formulation

We consider a scalar field $u(x,t)$ governed by

$$
\partial_t u = \mathcal{L}u + \mathcal{N}(u),
$$

on a periodic domain. The solution is approximated by a truncated Fourier series

$$
u(x,t) \approx \sum_{k=-K}^{K} \widehat{u}_k e^{ikx}.
$$

Applying the Fourier transform yields

$$
\partial_t \widehat{u}_k = \widehat{\mathcal{L}}_k \widehat{u}_k + \widehat{[\mathcal{N}(u)]}_k.
$$

The nonlinear term is evaluated in physical space, requiring forward and inverse FFTs at every time step.

# Software design

The software is implemented in MATLAB and uses the `spmd` programming model to enable parallel execution across multiple GPUs. The design follows two complementary strategies.

In the first strategy, a single multidimensional FFT is distributed across GPUs using domain decomposition. Each GPU computes local FFTs and participates in communication steps required to assemble the global transform.

In the second strategy, multiple physical fields are distributed across GPUs. This enables concurrent FFT computations and is particularly suitable for multiphysics models.

The implementation is modular and can be extended to other pseudo-spectral solvers beyond PFC models.

# Multi-GPU FFT strategies

## Distributed FFT

A three-dimensional FFT is decomposed across GPUs using slab decomposition. Each GPU computes local transforms, followed by communication and global assembly.

The PFC free energy is given by

$$
F[\psi] = \int_\Omega \left[
\frac{\psi}{2}(\varepsilon + \mathcal{L})\psi + \frac{\psi^4}{4}
\right] d\mathbf r,
$$

with evolution equation

$$
\partial_t \psi = (\varepsilon + \mathcal{L}) \nabla^2 \psi + \nabla^2 \psi^3.
$$

## Multiphysics coupling

For multiphysics PFC models, fields are distributed across GPUs. The hydrodynamic PFC system reads

$$
\begin{aligned}
\partial_t \psi &= \nabla^2\left(\frac{\delta F[\psi]}{\delta \psi}\right) - \mathbf{v} \cdot \nabla \psi, \\
\rho \partial_t \mathbf{v} &= \Gamma \nabla^2 \mathbf{v}
- \left\langle \psi \nabla \frac{\delta F[\psi]}{\delta \psi} \right\rangle,
\end{aligned}
$$

with Gaussian convolution

$$
\langle f \rangle(\mathbf{r}) = \int_\Omega
\frac{f(\mathbf{r}')}{(2\pi a_0^2)^{3/2}}
\exp\left(-\frac{|\mathbf{r}-\mathbf{r}'|^2}{2 a_0^2}\right)
d\mathbf{r}'.
$$

# Implementation

Example for distributed FFT:

```matlab
spmd
    for n = 1:n_timeSteps
        psi_hat = fftn(psi);
        nonlinear = fftn(psi.^3);
        psi_hat = psi_hat + dt * (linear_operator .* psi_hat + nonlinear);
        psi = ifftn(psi_hat);
    end
end
````

Example for multiphysics:

```matlab
spmd
    for n = 1:n_timeSteps
        if spmdIndex == 1
            psi_hat = fftn(psi);
        elseif spmdIndex == 2
            v_hat = fftn(v);
        end
    end
end
```

# Acknowledgements

We acknowledge support from the Deutsche Forschungsgemeinschaft (DFG), project numbers 447241406 and 493401063. We also acknowledge computing resources provided by the NHR Center at TU Dresden.

# AI usage disclosure

No generative AI tools were used in the development of this software or the preparation of this manuscript.

# References

```
```

