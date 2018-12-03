
def traj_distr_kl(new_mu, new_sigma, new_traj_distr, prev_traj_distr, tot=True):
    """
    Compute KL divergence between new and previous trajectory
    distributions.

    :param new_mu: T x dX, mean of new trajectory distribution.
    :param new_sigma: T x dX x dX, variance of new trajectory distribution.
    :param new_traj_distr: A linear Gaussian policy object, new
                           distribution.
    :param prev_traj_distr: A linear Gaussian policy object, previous
                            distribution.

    :returns kl_div: The KL divergence between the new and previous
                     trajectories.
    """
    T, dX, dU = new_mu.shape[0], new_traj_distr.dX, new_traj_distr.dU
    kl_div = np.zeros(T)

    for t in range(T):
        K_prev = prev_traj_distr.K[t, :, :]
        K_new = new_traj_distr.K[t, :, :]

        k_prev = prev_traj_distr.k[t, :]
        k_new = new_traj_distr.k[t, :]

        sig_prev = prev_traj_distr.pol_covar[t, :, :]
        sig_new = new_traj_distr.pol_covar[t, :, :]

        chol_prev = prev_traj_distr.chol_pol_covar[t, :, :]
        chol_new = new_traj_distr.chol_pol_covar[t, :, :]

        inv_prev = prev_traj_distr.inv_pol_covar[t, :, :]
        inv_new = new_traj_distr.inv_pol_covar[t, :, :]

        logdet_prev = 2 * sum(np.log(np.diag(chol_prev)))
        logdet_new = 2 * sum(np.log(np.diag(chol_new)))

        K_diff, k_diff = K_prev - K_new, k_prev - k_new
        mu, sigma = new_mu[t, :dX], new_sigma[t, :dX, :dX]

        # TODO(jhoang): link to some math that explains this. The idea is we can derive a closed-form of the expected 
        # value of KL divergence between two linear time varying Gaussian.
        kl_div[t] = max(
                0,
                0.5 * (logdet_prev - logdet_new - new_traj_distr.dU +
                       np.sum(np.diag(inv_prev.dot(sig_new))) +
                       k_diff.T.dot(inv_prev).dot(k_diff) +
                       mu.T.dot(K_diff.T).dot(inv_prev).dot(K_diff).dot(mu) +
                       np.sum(np.diag(K_diff.T.dot(inv_prev).dot(K_diff).dot(sigma))) +
                       2 * k_diff.T.dot(inv_prev).dot(K_diff).dot(mu))
        )

    return np.sum(kl_div)


class LQR(object):

    def forward(self):
        pass

    def compute_costs(self):
        pass

    def update(self, eta, traj_distr, kl_mult):
        kl_step = self.base_kl_step * step_mult

        for itr in range(self.max_iters):
            traj_distr = self.backward(prev_traj_distr, dynamics)

            new_mu, new_sigma = self.forward(traj_distr, dynamics)
            kl_div = traj_distr_kl(new_mu, new_sigma, traj_distr, prev_traj_distr)
        
            diff = kl_div - kl_step
            if abs(diff) < 0.1 * kl_step:
                break
            
            if diff < 0:
                # Eta was too big.
                max_eta = eta
                geom = np.sqrt(min_eta * max_eta)  # Geometric mean.
                new_eta = max(geom, 0.1 * max_eta)
            else: 
                # Eta was too small.
                min_eta = eta
                geom = np.sqrt(min_eta * max_eta)  # Geometric mean.
                new_eta = min(geom, 10.0 * min_eta)

        return traj_distr, eta

    def backward(self, prev_traj_distr, dynamics):
        """Performs LQR backward pass. This computes a new linear Gaussian
        policy object.

        :param prev_traj_distr: A linear Gaussian policy object from previous
                                iteration.
        :returns traj_distr: A new linear Gaussian policy object.
        """
        T = prev_traj_distr.T
        dU = prev_traj_distr.dU
        dX = prev_traj_distr.dX
        
        # Pull out dynamics.
        Fm = dynamics.Fm
        fv = dynamics.fv

        # Allocate the value functions and state-value functions.
        Vxx = np.zeros((T, dX, dX))
        Vx = np.zeros((T, dX))
        Qtt = np.zeros((T, dX+dU, dX+dU))
        Qt = np.zeros((T, dX+dU))
            
        traj_distr = prev_traj_distr.copy()
        new_chol_pol_cov = np.zeros((T, dU, dU))
        
        idx_x = slice(dX)
        idx_u = slice(dX, dX+dU)

        # Cm is T x (dX + dU) x (dX + dU) the quadratic component of the cost.
        # cv is T x (dX + dU) the linear component of the cost.
        Cm, cv = self.compute_costs()
        
        for t in range(T - 1, -1, -1):
            # Qxuxu = Cxuxu + fxu.Vxx.fxu
            # Qxu = cxu + fxu.Vx(t+1) + fxu.
            Qtt[t] = Cm[t, :, :]  # (X+U) x (X+U)
            Qt[t] = cv[t, :]  # (X+U) x 1

            if t < T - 1:
                Qtt[t] += Fm[t, :, :].T.dot(Vxx[t+1, :, :]).dot(Fm[t, :, :])
                Qt[t] += Fm[t, :, :].T.dot(Vx[t+1, :] + Vxx[t+1, :, :].dot(fv[t, :]))
            # Symmetrize quadratic component.
            Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

            Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + Qtt[t, idx_x, idx_u].dot(traj_distr.K[t, :, :])
            Vx[t, :] = Qt[t, idx_x] + Qtt[t, idx_x, idx_u].dot(traj_distr.k[t, :])
                    
            # Compute Cholesky decomposition of Q function action
            # component. LL^T = Qtt
            U = sp.linalg.cholesky(Qtt[t, idx_u, idx_u])
            L = U.T

            # Compute mean terms.
            # k = -Quu^(-1)Qut -> k = -(LU)^(-1)Qut -> U k = -L^(-1)Qut
            new_k[t, :] = -sp.linalg.solve_triangular(
                U, sp.linalg.solve_triangular(L, Qt[t, idx_u], lower=True)
            )
            # K = -Quu^(-1)Qux -> K = -(LU)^(-1)Qux -> U k = -L^(-1)Qux
            new_K[t, :, :] = -sp.linalg.solve_triangular(
                U, sp.linalg.solve_triangular(L, Qtt[t, idx_u, idx_x], lower=True)
            )
            
            # x = U^(-1)L^(-1) = Qtt^(-1)
            Q_inv = sp.linalg.solve_triangular(
                U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
            )
            # TODO: why are why using cholesky matrix as the covariance and not Q_inv?
            new_chol_pol_cov[t, :, :] = sp.linalg.cholesky(Q_inv)

        traj_distr.K, traj_distr.k = new_K, new_k
        traj_distr.chol_pol_cov = new_chol_pol_cov

        return traj_distr

    def forward(self, traj_distr, dynamics):
        """Performs LQR forward pass. Computes state-action marginals from dynamics and policy.

        :param traj_distr: A linear Gaussian policy object.
        :param dynamics: TODO
        :returns mu: A T x (dX + dU) mean action vector
        :returns sigma: TODO
        """
        T = traj_distr.T
        dU = traj_distr.dU
        dX = traj_distr.dX
        
        idx_x = slice(dX)
        
        # Allocate space.
        sigma = np.zeros((T, dX+dU, dX+dU))
        mu = np.zeros((T, dX+dU))

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv
        dyn_covar = dynamics.dyn_covar

        # Set initial covariance.
        sigma[0, idx_x, idx_x] = traj_info.x0sigma
        mu[0, idx_x] = traj_info.x0mu
        
        for t in range(T):
            sigma[t, :, :] = np.vstack([
                np.hstack([
                    sigma[t, idx_x, idx_x],
                    sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)
                ]),
                np.hstack([
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(traj_distr.K[t, :, :].T) + \
                            traj_distr.pol_covar[t, :, :]
                ])
            ])
            mu[t, :] = np.hstack([
                mu[t, idx_x],
                traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]
            ])
            if t < T - 1:
                sigma[t+1, idx_x, idx_x] = Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + dyn_covar[t, :, :]
                mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
        return mu, sigma
