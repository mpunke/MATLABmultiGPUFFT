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
    orcid: 0000-0000-0000-0000
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

The first strategy distributes a single high-dimensional FFT across multiple GPUs using domain decomposition and inter-GPU communication, enabling computations beyond the memory capacity of individual GPUs. The second strategy distributes multiple coupled fields across GPUs, allowing concurrent FFT evaluations for multiphysics simulations.

The framework is motivated by and applied to phase-field crystal (PFC) models, which are governed by high-order partial differential equations and require fine spatial resolution in periodic domains. The implementation achieves significant speedups compared to CPU-based approaches, enabling large-scale simulations previously inaccessible within MATLAB.

# Statement of Need

The pseudo-spectral Fourier method is widely used for solving partial differential equations with periodic boundary conditions due to its high accuracy and efficiency. However, its computational performance is dominated by repeated multidimensional FFTs, which become a bottleneck for large-scale simulations.

Single-GPU implementations are limited by memory capacity and scalability. This limitation is particularly severe for phase-field crystal (PFC) models, which resolve atomic-scale structures over diffusive time scales and require large computational domains.

Existing approaches either rely on CPU-based parallelization or single-GPU acceleration, leaving a gap for accessible multi-GPU solutions within MATLAB. This work addresses this gap by providing the first MATLAB-based implementation of distributed multi-GPU FFT strategies tailored for pseudo-spectral solvers.

The need for such an approach is further emphasized in multiphysics extensions of PFC models, where multiple coupled fields significantly increase memory requirements and computational complexity.

# Mathematical Formulation

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

# Multi-GPU FFT Strategy

## Single FFT Distributed Across GPUs

A three-dimensional FFT of size $N_x \times N_y \times N_z$ is decomposed across multiple GPUs using slab decomposition. Each GPU performs local FFTs followed by communication steps to assemble the global transform.

The phase-field crystal free energy is given by

$$
F[\psi] = \int_\Omega \left[
\frac{\psi}{2}(\varepsilon + \mathcal{L})\psi + \frac{\psi^4}{4}
\right] d\mathbf r,
$$

with evolution equation

$$
\partial_t \psi = (\varepsilon + \mathcal{L}) \nabla^2 \psi + \nabla^2 \psi^3.
$$

## Multi-GPU Strategy for Multiphysics Models

For multiphysics PFC models, different fields are distributed across GPUs. We consider the hydrodynamic PFC system:

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

The implementation is based on MATLAB's `spmd` construct for parallel GPU execution.

Example for distributed FFT computation:

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

Example for multiphysics coupling:

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

# Availability

* Repository: https://github.com/mpunke/MATLABmultiGPUFFT
* License: CC-BY 4.0
* Archive: https://doi.org/10.5281/zenodo.18670913

# Acknowledgements

We acknowledge support from the Deutsche Forschungsgemeinschaft (DFG), project numbers 447241406 and 493401063. We also acknowledge computing resources provided by the NHR Center at TU Dresden.

# References

```
```
