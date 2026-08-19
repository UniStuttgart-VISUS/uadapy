import numpy as np
import scipy.stats as stats
from uadapy import Distribution

class IndependentJoint:
    """
    Joint Distributiuon of independent continuous distributions. 
    This class allows to combine multiple independent distributions into a single multivariate joint distribution.
    Univariate as well as multivariate distributions can be joined. 
    The resulting joint distribution will have a dimensionality equal to the sum of the dimensionalities of the individual distributions.

    Example usage::

        import scipy.stats as stats
        from uadapy.distributions import IndependentJoint
        a = stats.Normal()
        b = stats.t(5)
        j = IndependentJoint([a, b])

    """

    def __init__(self, distributions, permutation=None):
        if not isinstance(distributions, list) or len(distributions) < 2:
            raise ValueError("distributions must be a list of at least 2 Distribution objects or Distribution wrappable objects")
        self.distributions = [d if isinstance(d, Distribution) else Distribution(d) for d in distributions]
        self.dim = sum(d.n_dims for d in self.distributions)
        self.dim_permutation = np.asarray(permutation) if permutation is not None else np.eye(self.dim,dtype=int)
        if len(self.dim_permutation.shape) == 1:
            self.dim_permutation = np.eye(self.dim,dtype=int)[:,self.dim_permutation.astype(int)]
        if self.dim_permutation.sum() != self.dim or np.any(self.dim_permutation.sum(axis=1) != 1) or np.any(self.dim_permutation.sum(axis=0) != 1):
            raise ValueError("permutation must be a valid permutation of the dimensions of the joint distribution")
    
    def sample(self, n, seed=None):
        samples = []
        for d in self.distributions:
            s = d.sample(n, seed=seed)
            samples.append(s if len(s.shape) > 1 else s[:,None])
        return np.hstack(samples) @ self.dim_permutation
    
    def cov(self):
        covs = [d.cov() for d in self.distributions]
        cov = np.zeros((self.dim, self.dim))
        idx = 0
        for c in covs:
            dim = 1 if np.isscalar(c) else c.shape[0]
            cov[idx:idx+dim, idx:idx+dim] = c
            idx += dim
        return self.dim_permutation.T @ cov @ self.dim_permutation
    
    def pdf(self, x):
        idx = 0
        if len(x.shape) == 1:
            x = x[None, :]
        pdfs = np.ones(x.shape[0])
        x = x @ self.dim_permutation.T
        for d in self.distributions:
            dim = d.n_dims
            x_ = x[:, idx:idx+dim]
            if dim == 1:
                x_ = x_.ravel() # flatten to 1D array for univariate distributions
            if len(x_) == 1:
                x_ = x_.item() # convert to scalar if univariate and single sample
            p = d.pdf(x_)
            pdfs *= p
            idx += dim
        return pdfs
    
    def mean(self):
        m = np.zeros(self.dim)
        idx = 0
        for d in self.distributions:
            dim = d.n_dims
            m[idx:idx+dim] = d.mean()
            idx += dim
        return m @ self.dim_permutation
    
    def marginal(self, dims):
        dims = np.asarray(dims)
        dim2dist, dist2dim = self.__dim_dist_luts__()
        order = np.argmax(self.dim_permutation, axis=0)
        order_reverse = np.argmax(self.dim_permutation, axis=1)
        if dims.size == 1:
            dim = dims.item()
            dim_idx = order_reverse[dim]
            dist_idx = dim2dist[dim_idx]
            if dist2dim[dist_idx][0] == dist2dim[dist_idx][1] - 1: # univariate distribution
                return self.distributions[dist_idx]
            else: # multivariate distribution
                return self.distributions[dist_idx].marginal(dim_idx - dist2dim[dist_idx][0])
        else:
            dims_in_order = order_reverse[dims]
            # find all distributions that contain the requested dimensions
            requested_dist_per_dim = dim2dist[dims_in_order]
            requested_dim_in_dist = [dims_in_order[i] - dist2dim[requested_dist_per_dim[i]][0] for i in range(len(dims_in_order))]
            # group the requested dimensions by distribution
            dist_dims = {}
            for i in range(len(dims_in_order)):
                dim_idx = dims_in_order[i]
                dist_idx = requested_dist_per_dim[i]
                if dist_idx not in dist_dims:
                    dist_dims[dist_idx] = []
                dist_dims[dist_idx].append(requested_dim_in_dist[i])
            # extract the required marginals
            marginals = {}
            for dist_idx, dims in dist_dims.items():
                d = self.distributions[dist_idx]
                marginals[dist_idx] = d.marginal(dims) if dist2dim[dist_idx][0] != dist2dim[dist_idx][1] - 1 else d
            # check if only one distribution object is remaining, then we can return it directly, otherwise we need to create a new IndependentJoint object
            if len(marginals) == 1:
                return list(marginals.values())[0]
            else:
                distidx2marginalidx = {dist_idx: i for i, dist_idx in enumerate(sorted(marginals.keys()))}
                marginals = [marginals[dist_idx] for dist_idx in sorted(marginals.keys())]
                marginal_ndims = [len(dist_dims[dist_idx]) for dist_idx in sorted(dist_dims.keys())]
                cumulative_ndims = np.cumsum([0] + marginal_ndims)[:-1]
                new_order = []
                i_d = np.array([0]*len(marginals))
                for i in range(len(dims_in_order)):
                    dist_idx = requested_dist_per_dim[i]
                    marginal_idx = distidx2marginalidx[dist_idx]
                    resulting_idx = cumulative_ndims[marginal_idx] + i_d[marginal_idx]
                    new_order.append(resulting_idx)
                    i_d[marginal_idx] += 1
                return IndependentJoint(marginals, permutation=new_order)
        
    
    def __dim_dist_luts__(self):
        dim2dist = []
        dist2dim = []
        idx = 0
        for i, d in enumerate(self.distributions):
            dim = d.n_dims
            dim2dist.extend([i]*dim)
            dist2dim.append((idx, idx+dim))
            idx += dim
        return np.array(dim2dist), np.array(dist2dim)