==================================================
UADAPy - Uncertainty-aware Data Analysis in Python
==================================================
UADAPy is a Python library to support an easy analysis of uncertain data. 
Here you find the most important information to get started.

.. toctree::
   :maxdepth: 1

   changelog.rst
   installation.rst
   examples.rst

Supported Methods
=================
UADAPy implements the following uncertainty-aware methods.

.. list-table::
   :header-rows: 1
   :widths: 25 25 30

   * - Method
     - Description
     - Reference
   * - :mod:`UAPCA <uadapy.dr.uapca>`
     - Uncertainty-aware Principal Component Analysis, propagating distribution parameters through a PCA projection.
     - Görtler et al., `Uncertainty-Aware Principal Component Analysis <https://doi.org/10.1109/TVCG.2019.2934812>`_, TVCG 2020
   * - :mod:`UAPCA Revisited <uadapy.dr.uapca_revisited>`
     - Sampling-based revisited variant of UAPCA that projects samples instead of propagating distribution parameters directly.
     - Friesecke et al., `Uncertainty-Aware PCA Revisited <https://doi.org/10.1109/TVCG.2025.3633868>`_, TVCG 2026
   * - :mod:`VIPurPCA <uadapy.dr.vipurpca>`
     - Visualizing and propagating uncertainty in PCA using automatic differentiation.
     - Zabel et al., `VIPurPCA: Visualizing and propagating uncertainty in principal component analysis. <https://doi.org/10.1109/TVCG.2023.3345532>`_, TVCG 2024
   * - :mod:`WGMM-UAPCA <uadapy.dr.wgmm_uapca>`
     - Weighted Gaussian Mixture Model extension of UAPCA for multimodal uncertain data.
     - Klötzl et al., `Uncertainty-Aware PCA for Arbitrarily Distributed Data Modeled by Gaussian Mixture Models <https://doi.org/10.1109/UncertaintyVisualization68947.2025.00010>`_, 2025 Workshop on Uncertainty Visualization
   * - :mod:`UAMDS <uadapy.dr.uamds>`
     - Uncertainty-aware Multidimensional Scaling for projecting normal distributions to lower dimensions.
     - Hägele et al., `Uncertainty-Aware Multidimensional Scaling <https://doi.org/10.1109/TVCG.2022.3209420>`_, TVCG 2023
   * - :mod:`UASTL <uadapy.temporal.uastl>`
     - Uncertainty-aware Seasonal-Trend Decomposition based on Loess for time series data.
     - Krake et al., `Uncertainty-Aware Seasonal-Trend Decomposition based on Loess <https://doi.org/10.1109/TVCG.2024.3364388>`_, TVCG 2025
   * - :mod:`Uncertainty-aware Fourier Transformation <uadapy.temporal.spectralAnalysis>`
     - Uncertainty-aware Fourier Transformation for time series data.
     - Evers et al., `Uncertainty-aware spectral visualization <https://doi.org/10.1109/TVCG.2025.3542898>`_, TVCG 2025

Classes
=======
In the following, we describe the most important data structure and provide detailed explanations on some concepts. 
This section is currently work in progress and will be extended over time.

.. toctree::
   :maxdepth: 1

   distribution.rst
   timeseries.rst

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
