Multi-GPU fast Fourier transforms in MATLAB (for large-scale phase-field crystal simulations)
======================================================

.. image:: https://img.shields.io/pypi/v/pyphot.svg
    :target: https://pypi.org/project/pyphot/

.. image:: https://zenodo.org/badge/70060728.svg
   :target: https://zenodo.org/badge/latestdoi/70060728

.. image:: https://static.pepy.tech/badge/pyphot
   :target: https://pepy.tech/project/pyphot

.. image:: https://static.pepy.tech/badge/pyphot/month
   :target: https://pepy.tech/project/pyphot

.. image:: https://img.shields.io/badge/python-3.9,_3.10,_3.11,_3.12,_3.13,_3.14-blue.svg

.. image:: https://joss.theoj.org/papers/10.21105/joss.08814/status.svg
   :target: https://doi.org/10.21105/joss.08814

We present a MATLAB-based framework for fast Fourier transforms on multiple GPUs for large-scale numerical simulations using the pseudospectral Fourier method. The software implements two complementary multi-GPU strategies that overcome single-GPU memory limitations and accelerate spectral solvers. This approach is motivated by and applied to phase-field crystal (PFC) models, which are governed by tenth-order partial differential equations, require fine spatial resolution, and are typically formulated in periodic domains. Our resulting numerical framework achieves significant speedups, approximately sixfold for standard PFC simulations and up to sixtyfold for multiphysics extensions, compared to a purely CPU-based implementation running on hundreds of cores.

.. image:: https://mybinder.org/badge.svg
  :target: https://mybinder.org/v2/gh/mfouesneau/pyphot/master?filepath=examples%2FQuickStart.ipynb

.. image:: https://img.shields.io/badge/render%20on-nbviewer-orange.svg
  :target: https://nbviewer.jupyter.org/github/mfouesneau/pyphot/tree/master/examples/


Running the code
------------
Download the repository, make sure NVIDIA GPUs are available (at least two), then run the corresponding matlab scripts:

* For a 2D pseudo-spectral solver of the Phase-Field Crystal Equation (underlying triangular symmetry), resulting in a dendritic growth morphology as visualized in the Figure 2(a), change directory to singleFFT_multiGPU_2D and run in MATLAB (at least two and maximal eight NVIDIA GPUs with latest driver updates must be available):

.. code::

  singleFFT_multiGPU_2D.m

* For a 3D pseudo-spectral solver of the Phase-Field Crystal Equation (underlying fcc symmetry), resulting in polycrystalline coarsening, change directory to singleFFT_multiGPU_3D and run in MATLAB (at least two and maximal eight NVIDIA GPUs with latest driver updates must be available):

.. code::

  singleFFT_multiGPU_3D.m


* For a 3D pseudo-spectral solver of the hydrodynamic Phase-Field Crystal Equation (underlying fcc symmetry) with additional modeling of elastoc relaxation, resulting in polycrystalline coarsening with as visualized in the Figure 2(b), change directory to multiGPU_multiPhysics_3D and run in MATLAB (four NVIDIA GPUs with latest driver updates must be available):


.. code::

  multiGPU_multiPhysics_3D.m
  



Contributors
------------

Author:

Maik Punke

Direct contributions to the code base:

* Maik Punke (@mpunke)
* Marco Salvalaglio (@lancon)

Related projects
----------------

- 3ms gitlab repository 
https://gitlab.com/3ms-group/
