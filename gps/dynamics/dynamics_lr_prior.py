
import numpy as np

from gps.dynamics.dynamics import Dynamics
from gps.utils.gaussian import extract_policy_using_fitted_gaussian


class DynamicsLRPrior(Dynamics):
    """
    Dynamics with linear regression, with arbitrary prior.
    """
    def __init__(self):
        Dynamics.__init__(self)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = self._hyperparams['prior']['type'](self._hyperparams['prior'])

    def update_prior(self, samples):
        """ Update dynamics prior. """
        X = samples.get_X()
        U = samples.get_U()
        self.prior.update(X, U)

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior

    def fit(self, X, U):
        """Fit the dynamics.

        :X: The states, dimension: N (batch_size) x T (horizon) x dX (state dimension).
        :U: The actions, dimesnsion: N (batch_size) x T (horizon) x dU (action dimension).
        :returns: Fm - linear coefficient, 
                  fv - constant coefficient, 
                  dyn_covar - the covariance (TODO: of what?).
        """
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval(dX, dU, Ys)
            sig_reg = np.zeros((dX+dU+dX, dX+dU+dX))
            sig_reg[it, it] = self._hyperparams['regularization']
            Fm, fv, dyn_covar = extract_policy_using_fitted_gaussian(
                    Ys, mu0, Phi, mm, n0, dwts, dX+dU, dX, sig_reg)
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar
        return self.Fm, self.fv, self.dyn_covar
