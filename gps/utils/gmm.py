import numpy as np
import scipy.linalg


def logsum(vec, axis=0, keepdims=True):
    maxv = np.max(vec, axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0


class GMM(object):

    def init_vars(self, N, D, K):
        self.sigma = np.zeros((K, D, D))
        self.mu = np.zeros((K, D))
        self.logmass = np.log(1.0 / K) * np.ones((K, 1))
        self.mass = (1.0 / K) * np.ones((K, 1))

        cidx = np.random.randint(0, K, size=(1, N))
        for i in range(K):
            cluster_idx = (cidx == i)[0]
            mu = np.mean(data[cluster_idx, :], axis=0)
            diff = (data[cluster_idx, :] - mu).T
            sigma = (1.0 / K) * (diff.dot(diff.T))
            self.mu[i, :] = mu
            self.sigma[i, :, :] = sigma + np.eye(D) * 2e-6

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
        logprobs = -0.5 * np.ones((N, K)) * D * np.log(2 * np.pi)
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
            logprobs[:, i] -= 0.5 * np.sum(soln**2, axis=0)

        logprobs += self.logmass.T
        return logprobs

    def mstep(self, logprobs, data, K):
        """Maximization step of EM algorithm.

        :param logprobs: The log likelihoods obtained from E-step.
        :param data: The data to fit.
        :param K: Number of clusters.
        """
        N, D = data.shape[0], data.shape[1]

        # Renormalize to get cluster weights.
        logw = logprobs - logsum(logprobs, axis=1)
        assert logw.shape == (N, K)

        # Renormalize again to get weights for refitting clusters.
        logwn = logw - logsum(logw, axis=0)
        assert logwn.shape == (N, K)
        w = np.exp(logwn)

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
        assert wdata.shape == (N, K, D)
        for i in range(K):
            # Compute weighted outer product.
            XX = wdata[:, i, :].T.dot(wdata[:, i, :])
            mu = self.mu[i, :]
            self.sigma[i, :, :] = XX - np.outer(mu, mu)

            # regularization.
            sigma = self.sigma[i, :, :]
            self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + 1e-6 * np.eye(D)

    def update(self, data, K, max_iterations=100, init=False):
        """Run EM algorithm.
        """
        N, D = data.shape[0], data.shape[1]
        if init:
            self.init_vars(N, D, K)
        
        prevll = -float('inf')
        
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logprobs = self.estep(data)

            # Compute log-likelihood.
            ll = np.sum(logsum(logprobs, axis=1))

            if ll < prevll:
                # TODO: Why does log-likelihood decrease sometimes?
                break

            # Converged.
            if np.abs(ll-prevll) < 1e-5*prevll:
                break

            prevll = ll

            # M-step: update clusters means and covariances.
            self.mstep(logprobs, data, K)

