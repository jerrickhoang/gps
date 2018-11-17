
class GMM(object):

    def estep(self, X):
        """Expectation step of EM algorithm.

        :param X: a N x D array of points.
        :returns logprobs: a N x K array of log likelihoods
        """
        N, D = X.shape

        # For each sample i, the log probability of x_i is 
        # -0.5 * log(|det(sigma)|) + 0.5 * (x - mu).(sigma^-1).(x - mu) - D/2*log(2 * pi)
        K = self.sigma.shape[0]

        # -D/2 * log (2 * pi)
        logprobs = -0.5*np.ones((N, K))*D*np.log(2*np.pi)
        for i in range(K):
            mu, sigma = self.mu[i], self.sigma[i]
            L = scipy.linalg.cholesky(sigma, lower=True)
            # -0.5*log(|det(sigma)|) = -0.5*log(det(L)^2)
            # = -log(prod(diag(L))) = -sum(log(diag(L)))
            logprobs[:, i] -= np.sum(np.log(np.diag(L)))

            diff = (X - mu).T
            # soln = L^-1 (x - mu)
            soln = scipy.linalg.solve_triangular(L, diff, lower=True)
            # (x - mu)T.(sigma^-1).(x - mu)= (x - mu)T(L^-1)T.(L^-1)(x - mu)
            # = solnT.soln = soln ** 2
            logprobs[:, i] -= 0.5*np.sum(soln**2, axis=0)

        logprobs += self.logmass.T
        return logprobs

    def mstep(self):
        self.logmass = logsum(logw, axis=0).T
        self.logmass = self.logmass - logsum(self.logmass, axis=0)
        assert self.logmass.shape == (K, 1)
        self.mass = np.exp(self.logmass)
        # Reboot small clusters.
        w[:, (self.mass < (1.0 / K) * 1e-4)[:, 0]] = 1.0 / N
        # Fit cluster means.
        w_expand = np.expand_dims(w, axis=2)
        data_expand = np.expand_dims(data, axis=1)
        self.mu = np.sum(w_expand * data_expand, axis=0)
        # Fit covariances.
        wdata = data_expand * np.sqrt(w_expand)
        assert wdata.shape == (N, K, Do)
        for i in range(K):
            # Compute weighted outer product.
            XX = wdata[:, i, :].T.dot(wdata[:, i, :])
            mu = self.mu[i, :]
            self.sigma[i, :, :] = XX - np.outer(mu, mu)

            if self.eigreg:  # Use eigenvalue regularization.
                raise NotImplementedError()
            else:  # Use quick and dirty regularization.
                sigma = self.sigma[i, :, :]
                self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + \
                        1e-6 * np.eye(Do)

    def update(self):
        """Run EM algorithm.
        """
        prevll = -float('inf')
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logprobs = self.estep(data)

            # Compute log-likelihood.
            ll = np.sum(logsum(logprobs, axis=1))

            if ll < prevll:
                break

            if np.abs(ll-prevll) < 1e-5*prevll:
                # Converged.
                break

            prevll = ll

            # M-step: update clusters means and covariances.
            self.mstep()

