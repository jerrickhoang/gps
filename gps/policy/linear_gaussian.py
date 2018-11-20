import numpy as np
from gps.policy.policy import Policy


class LinearGaussianPolicy(Policy):
    """
    Time-varying linear Gaussian policy.
    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)
    """

    def __init__(self, K, k, chol_pol_cov):
        """
        :param K: T x dU x dX control matrix. For each timestep the control
                  vector is a dU x dX transformation that turns a state into
                  an action.
        :param k: T x dU the constant term of the linear transformation.
        :param chol_pol_cov: T x dU x dU Cholesky policy covariance.
        """
        Policy.__init__(self)
        # Notice this fixed horizon T.
        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        assert k.shape == (self.T, self.dU)
        assert chol_pol_cov.shape == (self.T, self.dU, self.dU)

        self.K = K
        self.k = k
        self.chol_pol_cov = chol_pol_cov
        

    def act(self, x, t, noise=None):
        """U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)
        TODO: prove this reparameterization trick. I think it should be obvious
        but would be good to write something down.

        :param x: T x dX state vector.
        :param t: Time step.
        :param noise: T x dU noise vector.
        """
        assert t < self.T
        u = self.K[t].dot(x) + self.k[t]
        # TODO: should we generate the noise in this function instead of taking
        #       the noise as input?
        u += self.chol_pol_cov[t].T.dot(noise)
        return u

