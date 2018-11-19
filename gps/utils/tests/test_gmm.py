
import numpy as np
from scipy.stats import multivariate_normal

from gps.utils.gmm import GMM


def main():
    gmm = GMM()
    
    mu1 = np.array([0., -1.])
    mu2 = np.array([2., 2.])
    sigma1 = np.array([[ 1. , -0.5], [-0.5,  1.5]])
    sigma2 = np.array([[ -2. , 1.], [-2,  1]])

    dat1 = np.random.multivariate_normal(mu1, sigma1, 500)
    dat2 = np.random.multivariate_normal(mu2, sigma2, 500)
    
    dat = np.vstack([dat1, dat2])
    gmm.update(dat, 2)
    print gmm.mu # Should be close to mu1 and mu2
    print gmm.sigma # Should be close to sigma1 and sigma2

if __name__ == "__main__":
    main()
