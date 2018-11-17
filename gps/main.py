
from gps.environment.trajectory import Trajectory
from gps.environment.trajectory_list import TrajectoryList
from gps.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.trajectory_optimizer.lqr import LQR


class GPS(object):

    def __init__(self):
        self.dynamics = DyanmicsLRPrior()
        self.traj_opt = LQR()

    def update_dynamics(self, trajectories):
        pass

    def init_env(self):
        return None

    def collect_sample(self, policy):
        env = self.init_env()
        s = env.reset()
        traj = Trajectory()
        for t in range(T):
            a = policy.act(s)
            s, r, done, _ = env.step(a)
            traj.set_X(s, t)
            traj.set_U(a, t)
        return traj

def main(num_iter=10, num_trajs_per_iter=10, num_inner_iter=10):
    gps = GPS()
    trajs = []
    # 1. Collect sample with current time-varying linear Gaussian policy and
    #    ground truth dynamics.
    #    Input: policy, gt dynamics.
    #    Output: trajectories.
    for i in range(num_iter):
        for j in range(num_trajs_per_iter):
            trajs.append(gps.collect_sample())
    trajs = TrajectoryList(trajs)

    # 2. Update the normal-inverse-wishart prior and use the prior to obtain the MAP
    #    estimate of the Gaussian dynamics.
    #    Input: trajectories.
    #    Output: updated prior and dynamics.
    gps.update_dynamics(trajs)

    # 3. Calculate the KL divergence step size by estimating costs under previous and
    #    current dynamics. See https://arxiv.org/pdf/1607.04614.pdf appendex A, B for details.
    #    Input: prev and current dynamics and policies.
    #    Output: step size adjustment variable.
    gps.update_kl_step_mult()

    for i in range(num_inner_iter):
        gps.update_traj_distr()
 

if __name__ == "__main__":
    main()
