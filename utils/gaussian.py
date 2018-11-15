
def gauss_fit_wishart_prior(pts, phi, m, mu0, n0, weights, dX, dU, sigma_reg):
    """Fit a guassian using MAP estimate and wishart prior.
    See http://www.jmlr.org/papers/volume17/15-522/15-522.pdf appendix A.3 for explanation

    :param pts: TODO: dimension
    :param phi: parameter of prior normal-wishart distribution TODO: what is the meaning of this parameter.
    :param m: variance coefficient - parameter of prior normal-wishart distribution.
    :param mu0: mean - parameter of prior normal-wishart distribution.
    :param n0: count - parameter of prior normal-wishart distribution.
    :param weights: weights of the points TODO: dimension.
    :sigma_reg: sigma regularization.

    :returns mu: mean of the fitted gaussian.
    :returns sigma: sigma of the fitted gaussian.
    """
    # Build weights matrix.
    D = np.diag(weights)
    
    # Compute empirical mean and covariance.
    emp_mean = np.sum((pts.T * dwts).T, axis=0)
    diff = points - empirical_mean
    emp_sigma = diff.T.dot(D).dot(diff)
    emp_sigma = 0.5 * (emp_sigma + emp_sigma.T)
    
    # MAP estimate of joint distribution.
    N = dwts.shape[0]
    mu = empirical_mean
    sigma = (N * emp_sigma + phi + (N * m) / (N + m) * np.outer(emp_mean - mu0, emp_mean - mu0)) / (N + n0)
    sigma = 0.5 * (sigma + sigma.T)
    
    # Add sigma regularization.
    sigma += sig_reg

    return sigma, mu
    
def extract_policy_from_gaussian(sigma, mu, dX, dU):
    """Conditioning to get the dynamics.
    
    :param mu: mean of the fitted gaussian.
    :param sigma: sigma of the fitted gaussian.
    :param dX: dimension of the state.
    :param dU: dimension of the action.
    
    :returns fd, fc, dynsig: TODO
    """
    # Conditioning to get dynamics.
    # TODO(jhoang): i don't quite understand this function
    fd = np.linalg.solve(sigma[:dX, :dX], sigma[:dX, dX:dX+dU]).T
    fc = mu[dX:dX+dU] - fd.dot(mu[:dX])
    dynsig = sigma[dX:dX+dU, dX:dX+dU] - fd.dot(sigma[:dX, :dX]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)
    return fd, fc, dynsig
