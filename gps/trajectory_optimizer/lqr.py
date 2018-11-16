

class LQR(object):

    def forward(self):
        pass

    def backward(self, prev_traj_distr):
        T = prev_traj_distr.T
        dU = prev_traj_distr.dU
        dX = prev_traj_distr.dX

        # Allocate the value functions and state-value functions.
        Vxx = np.zeros((T, dX, dX))
        Vx = np.zeros((T, dX))
        Qtt = np.zeros((T, dX+dU, dX+dU))
        Qt = np.zeros((T, dX+dU))
        
        for t in range(T - 1, -1, -1):
            pass

