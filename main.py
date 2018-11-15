
from gps.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.trajectory_optimizer.lqr import LQR


class GPS(object):

    def __init__(self):
        self.dynamics = DyanmicsLRPrior()
        self.traj_opt = LQR()

    def update_dynamics(self, trajectories):
        pass

    def collect_sample(self):
        pass


def main(num_iter=10, num_trajs_per_iter=10, num_inner_iter=10):
    gps = GPS()
    trajs = []
    for i in range(num_iter):
        for j in range(num_trajs_per_iter):
            trajs.append(gps.collect_sample())
    trajs = TrajectoryList(trajs)

    gps.update_dynamics(trajs)

    gps.update_kl_step_mult()

    for i in range(num_inner_iter):
        gps.update_traj_distr()
 

if __name__ == "__main__":
    main()
