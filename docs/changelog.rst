=========
Changelog
=========
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
