Multi-GPU fast Fourier transforms in MATLAB (for large-scale phase-field crystal simulations)
======================================================

.[![PyPI version](https://img.shields.io/pypi/v/optigob.svg)](https://pypi.org/project/optigob/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/optigob.svg)](https://pypi.org/project/optigob/)
[![Documentation Status](https://readthedocs.org/projects/optigob/badge/?version=latest)](https://optigob.readthedocs.io/en/latest/?badge=latest)

[GitHub](https://github.com/colmduff/OptiGob) |
[Documentation](https://optigob.readthedocs.io/en/latest/) |


We present a MATLAB-based framework for fast Fourier transforms on multiple GPUs for large-scale numerical simulations using the pseudospectral Fourier method. The software implements two complementary multi-GPU strategies that overcome single-GPU memory limitations and accelerate spectral solvers. This approach is motivated by and applied to phase-field crystal (PFC) models, which are governed by tenth-order partial differential equations, require fine spatial resolution, and are typically formulated in periodic domains. Our resulting numerical framework achieves significant speedups, approximately sixfold for standard PFC simulations and up to sixtyfold for multiphysics extensions, compared to a purely CPU-based implementation running on hundreds of cores.

.. image:: https://mybinder.org/badge.svg
  :target: https://mybinder.org/v2/gh/mfouesneau/pyphot/master?filepath=examples%2FQuickStart.ipynb

.. image:: https://img.shields.io/badge/render%20on-nbviewer-orange.svg
  :target: https://nbviewer.jupyter.org/github/mfouesneau/pyphot/tree/master/examples/


Running the code
------------
Download the repository, make sure NVIDIA GPUs are available (at least two), then run the corresponding matlab scripts:

* For a 2D pseudo-spectral solver of the Phase-Field Crystal Equation (underlying triangular symmetry), resulting in a dendritic growth morphology as visualized in the Figure 2(a), change directory to `singleFFT_multiGPU_2D` and run in MATLAB (at least two and maximal eight NVIDIA GPUs with latest driver updates must be available):

 ```matlab
  singleFFT_multiGPU_2D.m
```

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
* Marco Salvalaglio (@marcosalvalaglio)

Related projects
----------------

- 3ms gitlab repository 
https://gitlab.com/3ms-group/
