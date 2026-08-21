import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import _multivariate as mv


class Distribution:
    """
    The Distribution class provides a consistent interface to a variety of distributions.
    
    Attributes
    ----------
    model
        The underlying concrete distribution model, a `scipy.stats` distribution object or an array of samples
    name : str 
        Name of the distribution type, e.g. 'Normal'
    n_dims : int
        Dimensionality of the distribution
    """

    def __init__(self, model, name="", n_dims=1):
        """
        Creates a distribution, if samples are passed as the first parameter,
        no assumptions about the distribution are made. For the pdf and the sampling,
        a KDE is used. If the name is "Normal", the samples
        are treated as samples of a normal distribution.

        Parameters
        ----------
        model: 
            A scipy.stats distribution or samples
        name: str, optional
            The name of the distribution
        n_dims: int, optional
            The dimensionality of the distribution (default is 1)
        """
        if name:
            self.name = name
        else:
            self.name = model.__class__.__name__
        if isinstance(model, np.ndarray) and name == "Normal":
            mean = np.mean(model, axis=0)
            cov = np.cov(model, rowvar=False)
            self.model = stats.multivariate_normal(mean, cov, allow_singular=True)
        else:
            self.model = model
        mean = self.mean()
        if isinstance(mean, np.ndarray):
            self.n_dims = len(self.mean())
        else:
            self.n_dims = 1
        self.kde = None
        if isinstance(self.model, np.ndarray):
            self.kde = stats.gaussian_kde(self.model.T)


    def sample(self, n: int, seed = None) -> np.ndarray:
        """
        Creates samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples.
        seed : int | rng, optional
            Seed for downstream RNG, or specific RNG to be used. Default is None.

        Returns
        -------
        np.ndarray
            Samples of the distribution.
        """
        if isinstance(self.model, np.ndarray):
            return self.kde.resample(n, seed).T if self.n_dims > 1 else self.kde.resample(n, seed)
        if hasattr(self.model, 'sample') and callable(self.model.sample):
            return self.model.sample(n, seed=seed)
        if hasattr(self.model, 'rvs') and callable(self.model.rvs):
            return self.model.rvs(size=n, random_state=seed)
        if hasattr(self.model, 'resample') and callable(self.model.resample):
            return self.model.resample(size=n, seed=seed)


    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Computes the probability density function.

        Parameters
        ----------
        x : np.ndarray or float
            The position where the pdf should be evaluated.

        Returns
        -------
        np.ndarray or float
            Probability values of the distribution at the given sample points.
        """
        if isinstance(self.model, np.ndarray):
            return self.kde.pdf(x.T)
        if not hasattr(self.model, 'pdf'):
            raise AttributeError(f"The model has no pdf. {self.model.__class__.__name__}")
        else:
            return self.model.pdf(x)
    

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Computes the cumulative density function.

        Parameters
        ----------
        x : np.ndarray or float
            The position where the cdf should be evaluated.

        Returns
        -------
        np.ndarray or float
            Cumulative probability values of the distribution at the given sample points.
        """
        if isinstance(self.model, np.ndarray):
            raise AttributeError("CDF not implemented for sample-based distributions.")
        if not hasattr(self.model, 'cdf'):
            raise AttributeError(f"The model has no cdf. {self.model.__class__.__name__}")
        else:
            return self.model.cdf(x)


    def mean(self) -> np.ndarray | float:
        """
        Expected value of the distribution.

        Returns
        -------
        np.ndarray or float
            Expected value of the distribution.
        """
        if isinstance(self.model, np.ndarray):
            return np.mean(self.model, axis=0)
        if hasattr(self.model, 'mean'):
            if callable(self.model.mean):
                return self.model.mean()
            return self.model.mean
        if hasattr(self.model, 'loc'):
            return self.model.loc
        if hasattr(self.model, 'mu'):
            return self.model.mu
        else:
           raise AttributeError(f"Mean not implemented yet! {self.model.__class__.__name__}")


    def cov(self) -> np.ndarray | float:
        """
        Covariance of the distribution.

        Returns
        -------
        np.ndarray or float
            Covariance of the distribution.
        """
        if isinstance(self.model, np.ndarray):
            return np.cov(self.model.T) if self.n_dims > 1 else np.cov(self.model)
        if hasattr(self.model, 'cov'):
            if callable(self.model.cov):
                return self.model.cov()
            return self.model.cov
        if hasattr(self.model, 'covariance'):
            if callable(self.model.covariance):
                return self.model.covariance
            return self.model.covariance
        if hasattr(self.model, 'var'):
            if callable(self.model.var):
                return self.model.var()
            return self.model.var
        if hasattr(self.model, 'variance'):
            if callable(self.model.variance):
                return self.model.variance()
            return self.model.variance
        if isinstance(self.model, mv.multivariate_t_frozen):
            return self.model.shape * (self.model.df / (self.model.df - 2))
        raise AttributeError(f"Covariance not implemented yet! {self.model.__class__.__name__}")


    def skew(self) -> np.ndarray | float:
        """
        Skewness of the distribution.

        Returns
        -------
        np.ndarray or float
            Skewness of the distribution.
        """
        if isinstance(self.model, np.ndarray):
            return stats.skew(self.model)
        if hasattr(self.model, 'stats') and callable(self.model.stats):
            return self.model.stats(moments='s')
        if isinstance(self.model, mv.multivariate_t_frozen):
            return 0 # TODO: check why there is a specific case for multivariate t distribution
        if hasattr(self.model, 'skew'):
            if callable(self.model.skew):
                return self.model.skew()
            return self.model.skew
        if hasattr(self.model, 'skewness'):
            if callable(self.model.skewness):
                return self.model.skewness()
            return self.model.skewness
        raise AttributeError(f"Skew not implemented! {self.model.__class__.__name__}")


    def kurt(self) -> np.ndarray | float:
        """
        Kurtosis of the distribution.

        Returns
        -------
        np.ndarray or float
            Kurtosis of the distribution.
        """
        if isinstance(self.model, np.ndarray):
            return stats.kurtosis(self.model)
        if hasattr(self.model, 'stats') and callable(self.model.stats):
            return self.model.stats(moments='k')
        if isinstance(self.model, stats.multivariate_normal):
            return 0 # TODO: check why there is a specific case for multivariate normal distribution
        if hasattr(self.model, 'kurt'):
            if callable(self.model.kurt):
                return self.model.kurt()
            return self.model.kurt
        if hasattr(self.model, 'kurtosis'):
            if callable(self.model.kurtosis):
                return self.model.kurtosis()
            return self.model.kurtosis
        raise AttributeError(f"Kurtosis not implemented! {self.model.__class__.__name__}")
        

    def marginal(self, dims, kde=None, n_samples_kde=1000, noise_level=0.0, seed=None):
        """
        Marginalizes the distribution retaining the specified dimensions.

        Parameters
        ----------
        dims : int or list of int or ndarray
            The dimensions to retain in the marginal distribution.
        kde : str, optional
            If 'KDE' or 'kde', a KDE estimation of the marginal is performed. 
            If 'KDE?' or 'kde?', a fallback to KDE is performed if the model does not have a marginal method. 
            Default is None, and an error is raised if the model does not have a marginal method.
        n_samples_kde : int, optional
            Number of samples to use for the marginal estimation via KDE. Default is 1000.
        noise_level : float, optional
            Level of uniform noise to be applied to the samples before KDE estimation. Default is 0.0.
        seed : int, optional
            Seed for the random number generator for reproducibility. Default is None.
        """
        if self.n_dims == 1:
            raise IndexError("Cannot marginalize a 1D distribution. Use the sample method to get samples.")
        if not isinstance(dims, np.ndarray):
            dims = np.array([dims]).ravel()
        
        # check if KDE estimation of the marginal is requested
        if 'KDE' == kde or 'kde' == kde:
            samples = self.sample(n_samples_kde, seed=seed)
            samples += stats.uniform.rvs(loc=-noise_level/2, scale=noise_level, size=samples.shape, random_state=seed)
            # update key to be the indices before the 'KDE' keyword
            return Distribution(samples[:, dims if len(dims) > 1 else dims[0]], n_dims=len(dims))
        # lets check if fallback KDE flag 'KDE?' / 'kde?' is present

        # lets check the model if it has a `marginal` method
        if hasattr(self.model, 'marginal') and callable(self.model.marginal):
            return Distribution(self.model.marginal(dims if len(dims) > 1 else dims[0]), n_dims=len(dims))
        elif isinstance(self.model, np.ndarray):
            return Distribution(self.model[:, dims if len(dims) > 1 else dims[0]], n_dims=len(dims))
        elif isinstance(self.model, stats._multivariate.multivariate_normal_frozen):
            # in case scipy version too old, i.e. 'marginal' is missing, we do it ourselves
            return Distribution(_multivariate_normal_marginal_(dims, self.model))
        # fallback to KDE if specified
        elif 'KDE?' == kde or 'kde?' == kde:
            return self.marginal(dims, kde='KDE', n_samples_kde=n_samples_kde, noise_level=noise_level, seed=seed)
        else:
            raise NotImplementedError(f"Marginal distribution not implemented for {self.model.__class__.__name__}. You may want to use the 'KDE' flag to estimate the marginal distribution via KDE.")


    def __getitem__(self, key):
        """
        Allows indexing into the distribution's dimensions.
        This yields the corresponding marginal distribution.

        Parameters
        ----------
        key : int or slice or tuple or list
            Index or slice to access dimensions of the distribution.
            In case the underlying model does not provide a marginalization function, a KDE can be used to estimate the marginal distribution. 
            Then, the key needs to contain the keyword 'KDE' followed by the number of samples to use for the marginal estimation via KDE, 
            the level of noise to be applied, and the seed for the random number generator.
            Example: distrib[0, 1, 'KDE', 1000] or distrib[0, 1, 'KDE', 1000, 0.1, 42]
            Using the 'KDE?' or 'kde?' keyword will fallback to KDE if the underlying model does not provide a marginalization function.

        Returns
        -------
        Distribuition
            The marginal distribution corresponding to the specified dimensions. 
        """
        # turn key into a list if it is not already
        if isinstance(key, (int, np.int_)):
            key = [key]
        if isinstance(key, slice):
            start, stop, step = key.indices(self.n_dims)
            key = list(range(start, stop, step))
        if isinstance(key, tuple):
            key = list(key)
        # now check if key is list or ndarray
        if not isinstance(key, (list, np.ndarray)):
            raise TypeError("Invalid index type. Must be int, slice, list, tuple, or ndarray.")

        # check if key contains 'KDE' or 'kde' or 'KDE?' or 'kde?'. 
        # Then the argument following it is the number of samples to use for the marginal estimation via KDE
        # and if there is another argument, it is the level of noise to be applied, 
        # and if there is yet another argument, it is the seed for the random number generator
        kde_index = None
        for i, k in enumerate(key):
            if k in ['KDE', 'kde', 'KDE?', 'kde?']:
                kde_index = i
                break
        if kde_index is not None:
            kde_flag = key[kde_index]
            n_samples_kde = 1000
            noise_level = 0.0
            seed = None
            if len(key) > kde_index + 1:
                n_samples_kde = key[kde_index + 1]
            if len(key) > kde_index + 2:
                noise_level = key[kde_index + 2]
            if len(key) > kde_index + 3:
                seed = key[kde_index + 3]
            # remove the kde flag and its parameters from the key
            key = key[:kde_index]
            if len(key) == 1:
                key = key[0]
            return self.marginal(key, kde=kde_flag, n_samples_kde=n_samples_kde, noise_level=noise_level, seed=seed)
        return self.marginalize(key)


def _multivariate_normal_marginal_(dims, model):
    if len(dims) == 1:
        mean = model.mean[dims[0]]
        var = model.cov[dims[0], dims[0]]
        return stats.norm(loc=mean, scale=np.sqrt(var))
    else:
        mean = model.mean[dims]
        cov = model.cov[:,dims][dims,:]
        return stats.multivariate_normal(mean=mean, cov=cov)