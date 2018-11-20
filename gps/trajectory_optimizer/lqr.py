

class LQR(object):

    def forward(self):
        pass

    def compute_costs(self):
        pass

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

    def forward(self):
        pass

    
