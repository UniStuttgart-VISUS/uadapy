=========
Changelog
=========
0.2.0
---- 

* Breaking changes:

    * Remove parameter :code:`seed` from :code:`plot_contour`; positional arguments after :code:`quantiles` shift accordingly
    * Default quantiles in :code:`plot_contour` are now :code:`[25, 75, 95]`
    * :code:`plot_samples` and :code:`plot_contour` raise a :code:`ValueError` for distributions that are not 2-dimensional (previously, :code:`plot_samples` silently used the first two dimensions)

* New datasets: :code:`load_wine`, :code:`load_breast_cancer`, :code:`load_student_grades`, synthetic shape generators
* Optional parameter :code:`normal` for :code:`load_iris()`
* Distribution class:

    * Add :code:`cdf()`, :code:`marginal()`, and indexing for marginalization over dimensions
    * Support sampling from models that provide a :code:`sample()` method
    * Allow singular covariance matrices for normal distributions
    * Bug fix: covariance computation for 1D distributions

* Dimensionality reduction:

    * Optional parameter :code:`weights` for :code:`uapca()`, :code:`compute_ua_cov()`, :code:`compute_uapca()`, and :code:`transform_uapca()`
    * Add modules :code:`wgmm_uapca`, :code:`uapca_revisited`, and :code:`vipurpca`
    * Bug fix: indexing of covariance matrices in :code:`transform_uapca`

* Optional parameters :code:`point_size` and :code:`alpha` for :code:`plot_samples`, consistent with nD sample plots
* Reproducible jitter in strip plots of :code:`plot_1d_distribution` using :code:`seed`
* Bug fix: frequency computation in spectral analysis


0.1.0
---- 

* Reorganization of plotting interface: input parameters are now more consistent across different plotting functions, default parameters changed
* Set default colors for all the plots
* Add distribution colors and colorblind safe option for high dimensional plots
* Bug fix in default computation of ranges in contour plots
* Dynamic computation of ranges based on samples
* Support for time series analysis:

    * Time series plotting functions
    * Uncertainty-aware seasonal trend decomposition based on loess (UASTL)
    * Uncertainty-aware Fourier analysis
* Use pytest for regression tests
* Add Shepard plots for quality evaluation
* Bug fix: Align axes in SPLOMs

0.0.2
---- 

* Reorganization of imports
* Documentation of distribution class
* Hideable mean in violin plot
* Consistency of various plotting functions

    * plotting functions return created figure and axis objects
    * plotting functions have optional parameter :code:`showplot` resulting in display of the figure
    * same parameter naming
* controllable width for box and violin plot
* dedicated plotting functions for 1D distributions plots e.g. :code:`generate_boxplot(...)` 


0.0.1
---- 
Features in first version:

* Uncertainty-aware Principle Component Analysis
* Uncertainty-aware Multi-dimensional Scaling
* Basic visualizations:

    * N-dimensional distributions: scatterplot matrices, iso-lines of probability distribution
    * 1D distributions: box plots, violin plots, dot plots
