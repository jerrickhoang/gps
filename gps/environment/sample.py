import numpy as np


class Sample(object):
    """Class that handles the representation of a trajectory and stores a
    single trajectory.
    """
    def __init__(self):
        self.T = agent.T
        self.dX = agent.dX
        self.dU = agent.dU

        self._X = np.empty((self.T, self.dX))
        self._X.fill(np.nan)
        self._U = np.empty((self.T, self.dU))
        self._U.fill(np.nan)

    def invalidate(self):
        self._X.fill(np.nan)
        self._U.fill(np.nan)

    def set_X(self, X, t):
        self._X[t, :] = X

    def set_U(self, U, t):
        self._U[t, :] = U

    def get_Xs(self):
        return self._X
    
    def get_X(self, t):
        return self._X[t]

    def get_Us(self):
        return self._U

    def get_U(self, t):
        return self._U[t]
