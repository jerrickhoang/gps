from gps.utils.gmm import GMM


class DynamicsGMMPrior(object):
    """
    A dynamics prior encoded as a GMM over [x_t, u_t, x_t+1] points.
    See:
        S. Levine*, C. Finn*, T. Darrell, P. Abbeel, "End-to-end
        training of Deep Visuomotor Policies", arXiv:1504.00702,
        Appendix A.3.
    """
    def __init__(self, min_samples_per_cluster=?, max_clusters=?, max_samples=?):
        """
        :param min_samples_per_cluster: Minimum samples per cluster.
        :param max_clusters: Maximum number of clusters to fit.
        :param max_samples: Maximum number of trajectories to use for
                            fitting the GMM at any given time.
        :param strength: Adjusts the strength of the prior.
        """
        self.X = None
        self.U = None
        self.gmm = GMM()
        self._min_samples_per_cluster = min_samples_per_cluster
        self._max_samples = max_samples
        self._max_clusters = max_clusters
        self._strength = strength

    def initial_state(self):
        """ Return dynamics prior for initial time step. """
        # Compute mean and covariance.
        mu0 = np.mean(self.X[:, 0, :], axis=0)
        Phi = np.diag(np.var(self.X[:, 0, :], axis=0))

        # Factor in multiplier.
        n0 = self.X.shape[2] * self._strength
        m = self.X.shape[2] * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi = Phi * m
        return mu0, Phi, m, n0

    def update(self, X, U):
        """
        Update prior with additional data.
        :param X: A N x T x dX matrix of sequential state data.
        :param U: A N x T x dU matrix of sequential control data.
        """
        # Constants.
        T = X.shape[1] - 1

        # Append data to dataset.
        if self.X is None:
            self.X = X
        else:
            self.X = np.concatenate([self.X, X], axis=0)

        if self.U is None:
            self.U = U
        else:
            self.U = np.concatenate([self.U, U], axis=0)

        # Remove excess samples from dataset.
        start = max(0, self.X.shape[0] - self._max_samples + 1)
        self.X = self.X[start:, :]
        self.U = self.U[start:, :]

        # Compute cluster dimensionality.
        D = X.shape[2] + U.shape[2] + X.shape[2]  #TODO: Use Xtgt.

        # Create dataset.
        N = self.X.shape[0]
        xux = np.reshape(
            np.c_[self.X[:, :T, :], self.U[:, :T, :], self.X[:, 1:(T+1), :]],
            [T * N, D]
        )

        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samples_per_cluster))))
        LOGGER.debug('Generating %d clusters for dynamics GMM.', K)

        # Update GMM.
        self.gmm.update(xux, K)

    def eval(self, Dx, Du, pts):
        """Evaluate prior.
        :param pts: A N x Dx+Du+Dx matrix representing (x_t, u_t, x_t+1).
        :returns : TODO
        """
        # Construct query data point by rearranging entries and adding
        # in reference.
        assert pts.shape[1] == Dx + Du + Dx

        # Perform query and fix mean.
        mu0, Phi, m, n0 = self.gmm.inference(pts)

        # Factor in multiplier.
        n0 = n0 * self._strength
        m = m * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0
