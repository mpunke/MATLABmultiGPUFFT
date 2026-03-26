---
title: 'Multi-GPU fast Fourier transforms in MATLAB for large-scale phase-field crystal simulations'
tags:
  - MATLAB
  - fft
  - multi-gpu
  - fft
  - phase-field crystal
authors:
  - name: Maik Punke
    orcid: 0000-0002-3564-7942
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Marco Salvalaglio
    orcid: 0000-0002-4217-0951
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Institute of Scientific Computing, TU Dresden, 01062 Dresden, Germany
   index: 1
 - name: Dresden Center for Computational Materials Science, TU Dresden, 01062 Dresden, Germany
   index: 2
date: 26 March 2026
bibliography: paper.bib

---
\usepackage{listings}
# Summary

We present a MATLAB-based framework for two- and three-dimensional fast Fourier transforms on multiple GPUs for large-scale numerical simulations using the pseudo-spectral Fourier method. The software implements two complementary multi-GPU strategies that overcome single-GPU memory limitations and accelerate spectral solvers. This approach is motivated by and applied to phase-field crystal (PFC) models, which are governed by tenth-order partial differential equations, require fine spatial resolution, and are typically formulated in periodic domains. Our resulting numerical framework achieves significant speedups, approximately sixfold for standard PFC simulations and up to sixtyfold for multiphysics extensions, compared to a purely CPU-based implementation running on hundreds of cores.

# Statement of need

Large-scale simulations based on the pseudo-spectral Fourier method [@boyd2001chebyshev] are often limited by the memory capacity and performance of single GPUs due to their reliance on repeated multidimensional fast Fourier transforms (FFTs). To address this bottleneck, we present two complementary multi-GPU FFT strategies in two and three dimensions.

The first strategy distributes a single high-dimensional FFT across multiple GPUs via domain decomposition and inter-GPU communication, enabling Fourier transforms for problem sizes that exceed the memory of individual GPUs. The second strategy computes multiple FFTs concurrently by assigning different physical fields to separate GPUs with synchronized communication.

In the following, phase-field crystal (PFC) models [@Elder2002;@Elder2004;@Emmerich2012] serve as a representative application that underscores the need for such multi-GPU capabilities. PFC models resolve crystalline order at atomic length scales while evolving on large (diffusive) time scales, enabling the simulation of elasticity, defects, grain boundaries, and microstructure evolution within a unified mesoscale framework. Capturing these phenomena typically requires relatively large periodic domains but fine spatial resolution, leading to substantial memory demands and limiting the reachable size of single-GPU execution. This is even more relevant for multiphysics extensions that involve additional variables, such as coupled density/composition, velocity, and/or temperature fields, see, e.g., [@skogvoll2022hydrodynamic;@Punke_2022].

The pseudo-spectral Fourier method is a natural and widely adopted numerical approach for PFC and related models, as it allows for efficient and accurate evaluation of high-order spatial derivatives and convolution operators that arise in their governing equations, see, e.g., [@cheng2008numeric;@tegze2009advanced;@cheng2019energy;@Punke2023;@punke2025hybrid;@punke2026grain]. 
However, its performance is dominated by repeated multidimensional FFTs, which are required at every time step due to the nonlinear nature of the PDEs governing PFC dynamics, and quickly become the computational bottleneck at large scales. Efficient multi-GPU FFT strategies are therefore essential to fully exploit the accuracy and scalability of pseudo-spectral solvers.

As a representative use case, we thus exploit the developed strategies for multi-GPU execution in numerical simulations using a Fourier pseudo-spectral solver for the PFC model in two and three dimensions, where large spatial domains and coupled fields make single-GPU execution impractical. The algorithms are implemented in MATLAB, providing an accessible and extensible foundation for GPU-accelerated spectral simulations and introducing the first multi-GPU FFT implementation in MATLAB.


# Fourier pseudo-spectral method
We briefly review the basics of the Fourier pseudo-spectral method. Although the actual implementation is carried out in two and three dimensions, we restrict the following discussion to a one-dimensional setting for the sake of simplicity. We consider a generic evolution equation for a scalar field $u\equiv u(x,t)$ of the form
\begin{equation}\label{eq:realdyn}
\partial_t u = \mathcal{L}u + \mathcal{N}(u), \quad x \in [0,2\pi), \ t \ge 0,
\end{equation}
where $\mathcal{L}$ is a linear differential operator, and $\mathcal{N}(u)$ is a nonlinear term given by a polynomial in $u$.

On the periodic domain $[0,2\pi)$, we approximate $u(x,t)$ by a truncated Fourier series
\begin{equation}
u(x,t) \approx u_N(x,t) = \sum_{k=-K}^{K} \widehat{u}_k  e^{ikx},
\end{equation}
with $N = 2K+1$ modes  $\widehat{u}_k\equiv \widehat{u}(k,t)$ and equispaced collocation points
\begin{equation}
x_j = \frac{2\pi j}{N}, \quad j = 0,\dots,N-1.
\end{equation}
In the Fourier pseudo-spectral method, a time-dependent ordinary differential equation for $\widehat{u}_k$ is obtained by applying the Fourier transform to the left and right-hand side of Eq. \eqref{eq:realdyn}, reading
\begin{equation}\label{eq:fdyn}
\partial_t \widehat{u}_k = \widehat{\mathcal{L}}_k \widehat u_k + \widehat{[\mathcal{N}(u)]}_k \qquad \forall k=-K, \dots, K
\end{equation}
with $\widehat{\mathcal{L}}_k$ a polynomial in $k$ as $\widehat{[\partial^{n}_x u]}_k = (ik)^n \widehat{u}_k$, $\forall n\in \mathbb{N}$ while the nonlinear polynomial term is evaluated point-wise in physical space at every time iteration and then transformed to Fourier space. For a given $u(x,t)$ and time step $\Delta t$,  $u(x,t+\Delta t)$ is computed by first determining $\widehat{u}_k(t)$ and $\widehat{[\mathcal{N}(u)]}_k(t)$ via FFTs, solving the system of ODEs \eqref{eq:fdyn} using a suitable time-integration scheme, and then computing the inverse FFT from the resulting $\widehat{u}_k(t+\Delta t)$. The concept can be readily extended to higher spatial dimensions. More details can be found in Ref. [@boyd2001chebyshev].



# Single FFT on Multiple GPUs

Distributed multi-GPU FFTs based on domain decomposition and inter-GPU communication are well established in high-performance computing; see, e.g.,  [@pekurovsky2012p3dFFT;@ayala2020heFFTe;@verma2023scalable;@cuFFTmp]. These approaches enable scalable multi-dimensional FFTs on large GPU systems.
In contrast, FFT-based pseudo-spectral solvers for the PFC equation have so far been limited to single-GPU implementations  [@hallberg2025pypfc] or CPU-based parallelization  [@pinomaa2024openpfc;@skogvoll2024comfit]. To our knowledge, a dedicated multi-GPU FFT implementation tailored to large-scale three-dimensional PFC simulations has not been reported. Furthermore, a general multi-GPU FFT implementation is not yet available in MATLAB, independent of any specific application such as PFC simulations.

Following the strategy of  [@pekurovsky2012p3dFFT;@ayala2020heFFTe], we decompose a three-dimensional FFT of size $N_x \times N_y \times N_z$ across $G$ GPUs by slab decomposition along the $z$-direction, cf. Fig. \ref{fig:multiGPU}(a). Each GPU first performs local two-dimensional FFTs, then performs peer-to-peer (P2P) communication to redistribute the data, and finally performs a final one-dimensional FFT along the remaining dimension. After completion, each GPU holds a portion of the transformed Fourier-space array.

We apply this strategy to the PFC equation, which describes the evolution of a periodic density field $\psi\equiv \psi(\mathbf{x},t)$  [@Elder2002;@Elder2004]. The model builds on a free-energy functional, which, for example, can be expressed for face-centered cubic (FCC) crystal symmetry as
\begin{equation}
\label{eq:SHenergy}
F[\psi]
= \int_\Omega \left[
\frac{\psi}{2}\left(\varepsilon + \mathcal{L}\right)\psi
+ \frac{\psi^4}{4}
\right]  \mathrm{d}\mathbf r,
\end{equation}
with undercooling parameter $\varepsilon$ and  $\mathcal{L} = (1+\nabla^2)^2(4/3+\nabla^2)^2$ an operator describing spatial correlations within the  domain $\Omega$. The classical evolution equation reads
\begin{equation}
\label{eq:evolution}
\partial_t \psi
=\nabla^2 \dfrac{\delta F[\psi]}{\delta \psi}=(\varepsilon + \mathcal{L}) \nabla^2 \psi
+ \nabla^2 \psi^3.
\end{equation}
which retains the form of Eq. \eqref{eq:realdyn}. The equation is solved using a Fourier pseudo-spectral method with semi-implicit time integration  [@tegze2009advanced]. We note that alternative time-integration approaches can be considered  [@Punke2023].

We benchmark the multi-GPU solver for problem sizes ranging from $750^3$ to $1400^3$, achieving up to a sixfold speedup compared to a purely CPU-based implementation; see Fig. \ref{fig:multiGPU}(a). The benchmarks are performed on three systems from the HPC clusters provided by the NHR Center at TU Dresden. GPU computations are performed with four NVIDIA H100 SXM5 GPUs (94 GiB HBM2e each, HPC cluster Capella) and with eight NVIDIA A100 SXM4 GPUs (40 GiB HBM2 each, HPC cluster Alpha Centauri). CPU reference runs are performed on an Intel Xeon Platinum 8470 (100 cores, 2.00 GHz, HPC cluster Barnard). In Listing \ref{lst:matlab1}, the MATLAB implementation is illustrated.

Figure \ref{fig:multiGPU_examples}(a) illustrates dendritic solidification within the PFC framework, presented here as a representative two-dimensional benchmark example.

# Multiple GPU usage for Multiphysics PFC
The PFC framework readily supports multiphysics extensions. As an example, we consider the hydrodynamic phase-field crystal (hydrodynamic PFC) model  [@skogvoll2022hydrodynamic;@qiu2024grain] in three spatial dimensions, which augments the density field $\psi$ with a mesoscopic velocity field $\mathbf{v} \equiv (v_1(\mathbf{x},t),v_2(\mathbf{x},t),v_3(\mathbf{x},t))$:
\begin{equation}
\label{eq:hpfc}
\begin{aligned}
\partial_t \psi &= \nabla^2\left(\frac{\delta F[\psi]}{\delta \psi}\right) -\mathbf{v}  \cdot \nabla \psi, \\
\rho \partial_t \mathbf{v} &= \Gamma \nabla^2 \mathbf{v}
- \Big\langle \psi \nabla\frac{\delta F[\psi]}{\delta \psi} \Big\rangle ,
\end{aligned}
\end{equation}
with $\langle\ \cdot \ \rangle$ a local averaging obtained through a convolution with a Gaussian kernel 
\begin{equation}\label{eq:coarsegraining}
\langle \ \cdot\ \rangle(\mathbf{r})=\int_\Omega  \frac{ (\ \cdot\ )\left(\mathbf{r}^{\prime}\right)}{\left(2 \pi a_0^2\right)^{3 / 2}} \exp \left(-\frac{\left(\mathbf{r}-\mathbf{r}^{\prime}\right)^2}{2 a_0^2}\right) d \mathbf{r}^{\prime}.
\end{equation}
and $a_0$ the lattice spacing. Note that this operation just translates to a multiplication in the Fourier spectral method owing to the properties of the Fourier transform.

We distribute the fields across four GPUs, assigning one field per device and performing synchronized inter-GPU communication after each time step. This strategy enables handling problems larger than single-GPU memory limits. Benchmark results using four NVIDIA H100 GPUs demonstrate substantial runtime reductions compared to CPU execution on the previously described HPC clusters Capella and Barnard. In particular, for large computational domains where single-GPU simulations become infeasible, significant performance gains are observed; see Fig. \ref{fig:multiGPU}(b). Listing \ref{lst:matlab2} illustrates the MATLAB implementation. Figure \ref{fig:multiGPU_examples}(b) depicts polycrystalline coarsening within the hydrodynamic PFC framework, presented here as a representative three-dimensional benchmark case.


The adopted strategy of distributing different fields across multiple GPUs naturally extends to coarse-grained formulations based on complex amplitudes of the principal Fourier modes  [@salvalaglio2022coarse]. These models are well-suited to pseudo-spectral methods and typically require the simultaneous evolution of tens of coupled complex-valued fields, a setting for which we expect the present parallelization strategy to be particularly effective.


# Implementation Snippet

```python
%%initialize and decompose into slabs
% psi ...initial density field
% psiF...Fourier transformed initial density field
% lap ...discretized laplacian
% lin ...discretized linear operator (epsilon+L^2)nabla^2

spmd %parallel GPU session (G GPUs), on each GPU perform
    for  n=1:n_timeSteps %time iteration
        %%forward fftn
        psi = fft2(psi.^3); %fft2 along first two dimensions
        for i=1:G %P2P communication 
            psi_chunk = psi(localStart:localStart,:,:); %decompose each slab in a set of local chunks
            psi_chunk= spmdCat(psi_chunk,3,i); %stack the chunks together (along third dimension)
        end
        psi = psi_chunk;
        psi = fft(psi,[],3);  %fft along third dimension
        
        %%semi-implicit time step update
        psiF = (psiF+dt*lap.*psi)./(1-dt*lin); 

        %%backward ifftn
        psi = psiF;
        psi  = ifft(psi,[],3); %ifft along third dimension
        for i=1:G %P2P communication  
            psi_chunk = psi(:,:,localStart:localStart);%decompose each slab in a set of local chunks
            psi_chunk= spmdCat(psi_chunk,1,i); %stack the chunks together (along first dimension)
        end
        psi = psi_chunk;
        psi  = ifft2(psi); %ifft2 along first two dimensions
    end
end

%% stack solution 
psi = cat(3, gather(psi{:}));
```


```python
%%initialize on GPU1
% psi ...initial density field
% psiF...Fourier transformed initial density field
% v1  ...initial velocity field (first component)
% v2  ...initial velocity field (secod component)
% v3  ...initial velocity field (third component)
% lin ...discretized linear operator (epsilon+L^2)nabla^2
% k   ...discretized Fourier vector (first dimension)
% l   ...discretized Fourier vector (second dimension)
% m   ...discretized Fourier vector (third dimension)

%%initialize on GPU2...GPU4
% lap ...discretized laplacian
% k   ...discretized Fourier vector (first dimension)
% op  ...discretized linear operator (epsilon+L^2)
% cg  ... discretized convolution kernel
% additionally initialize on GPU2
% v1F ...Fourier transformed initial v1
% additionally initialize on GPU3
% v2F ...Fourier transformed initial v2
% additionally initialize on GPU4
% v3F ...Fourier transformed initial v3

spmd %parallel GPU session (4 GPUs)
    for  n=1:n_timeSteps %time iteration
        if spmdIndex==1 %GPU1 holding psi
            %%semi-implicit time step update of psi
            psiF = (psiF+dt*(lap.*fftn(psi.^3)-fftn(v1.*ifftn(k.*psiF)+v2.*ifftn(l.*psiF)+v3IF.*ifftn(m.*psiF))))./(1-dt*lin);
            psi = ifftn(psiF);
            %%send psi to GPU2...GPU4
            spmdSend(psi,[2:4],2); 
            %%receive v components from GPU2...GPU4
            v1 = spmdReceive(2,4);
            v2 = spmdReceive(3,5);
            v3 = spmdReceive(4,6);
            
        elseif spmdIndex==2 %GPU2 holding v1
            %receive psi from GPU1
            psi = spmdReceive(1,2);
            %%fully-implicit time step update of v1
            v1F = (v1F-dt/rho0 .*cg.*fftn(psi.*ifftn(k.*(fftn(psi.^3) + op.*fftn(psi)))))
                ./(1-deltatEuler/rho0.*.GammaS.*lap));
            v1 = ifftn(v1F);
            %send v1 to GPU1
            spmdSend(v1,1,4);
            
        elseif spmdIndex==3 %GPU3 holding v2
            %receive psi from GPU1
            psi = spmdReceive(1,2);
            %%fully-implicit time step update of v2
            v2F = (v2F-dt/rho0 .*cg.*fftn(psi.*ifftn(l.*(fftn(psi.^3) + op.*fftn(psi)))))
                ./(1-deltatEuler/rho0.*.GammaS.*lap));
            v2 = ifftn(v2F);
            %send v2 to GPU1
            spmdSend(v2,1,5);
            
        elseif spmdIndex==4 %GPU4 holding v3
            %receive psi from GPU1
            psi = spmdReceive(1,2);
            %%fully-implicit time step update of v3
            v3F = (v3F-dt/rho0 .*cg.*fftn(psi.*ifftn(m.*(fftn(psi.^3) + op.*fftn(psi)))))
                ./(1-deltatEuler/rho0.*.GammaS.*lap));
            v3 = ifftn(v3F);
            %send v3 to GPU1
            spmdSend(v3,1,6);
        end    
    end
end
```

# State of the field                                                                                                                  

Several tools exist for galactic dynamics computations:                                                     
`galpy` [@Bovy:2015] is a Python package with similar goals,
providing orbit integration and potential classes for galactic dynamics.                                                              
`NEMO` [@Teuben:1995] is a well-established, comprehensive stellar dynamics                                                           
toolbox written primarily in C, offering extensive functionality but with a                                                           
steeper learning curve and less integration with modern Python workflows.                                                             
Other tools like `GalPot` provide specific Milky Way potential models but lack                                                        
the broader dynamical analysis capabilities.                                                                                          
                                                                                                                                        
`Gala` was built rather than contributing to existing projects for several                                                            
reasons. First, `Gala` was designed from the ground up to integrate seamlessly                                                        
with the Astropy ecosystem, using `astropy.units` and `astropy.coordinates`                                                           
as core dependencies rather than optional features. This tight integration                                                            
enables natural workflows for astronomers already using Astropy. Second,                                                              
`Gala`'s object-oriented API with consistent interfaces across subpackages                                                            
(potentials, integrators, dynamics) provides a more modular and extensible                                                            
design than alternatives available at the time. Third, `Gala` fills a specific                                                        
niche between simple demonstration codes and full N-body simulation packages                                                          
like `Gadget` [@Springel:2005] – it focuses on the common tasks in galactic                                                             
dynamics research (orbit integration, potential evaluation, coordinate                                                                
transformations) while maintaining both performance through C implementations                                                         
and usability through its Python interface.  

# Software design

`Gala`'s design philosophy is based on three core principles: (1) to provide a
user-friendly, modular, object-oriented API, (2) to use community tools and
standards (e.g., Astropy for coordinates and units handling), and (3) to use
low-level code (C/C++/Cython) for performance while keeping the user interface
in Python. Within each of the main subpackages in `gala` (`gala.potential`,
`gala.dynamics`, `gala.integrate`, etc.), we try to maintain a consistent API
for classes and functions. For example, all potential classes share a common
base class and implement methods for computing the potential, forces, density,
and other derived quantities at given positions. This also works for
compositions of potentials (i.e., multi-component potential models), which
share the potential base class but also act as a dictionary-like container for
different potential components. As another example, all integrators implement a
common interface for numerically integrating orbits. The integrators and core
potential functions are all implemented in C without support for units, but the
Python layer handles unit conversions and prepares data to dispatch to the C
layer appropriately.Within the coordinates subpackage, we extend Astropy's
coordinate classes to add more specialized coordinate frames and
transformations that are relevant for Galactic dynamics and Milky Way research.

# Research impact statement

`Gala` has demonstrated significant research impact and grown both its user base
and contributor community since its initial release. The package has evolved
through contributions from over 18 developers beyond the original core developer
(@adrn), with community members adding new features, reporting bugs, and
suggesting new features.

While `Gala` started as a tool primarily to support the core developer's
research, it has expanded organically to support a range of applications across
domains in astrophysics related to Milky Way and galactic dynamics. The package
has been used in over 400 publications (according to Google Scholar) spanning
topics in galactic dynamics such as modeling stellar streams [@Pearson:2017],
Milky Way mass modeling, and interpreting kinematic and stellar population
trends in the Galaxy. `Gala` is integrated within the Astropy ecosystem as an
affiliated package and has built functionality that extends the widely-used
`astropy.units` and `astropy.coordinates` subpackages. `Gala`'s impact extends
beyond citations in research: Because of its focus on usability and user
interface design, `Gala` has also been incorporated into graduate-level galactic
dynamics curricula at multiple institutions.

`Gala` has been downloaded over 100,000 times from PyPI and conda-forge yearly
(or  2,000 downloads per week) over the past few years, demonstrating a broad
and active user community. Users span career stages from graduate students to
faculty and other established researchers and represent institutions around the
world. This broad adoption and active participation validate `Gala`'s role as
core community infrastructure for galactic dynamics research.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# AI usage disclosure

No generative AI tools were used in the development of this software, the writing
of this manuscript, or the preparation of supporting materials.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
