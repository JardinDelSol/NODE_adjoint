import torch
import numpy as np
from ode import odeint

# Hyper parameters
BATCH = 1
TIME = 10
DIM = 2


class Adjoint_ODE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, theta, odeint):
        traj = torch.zeros((BATCH, TIME, DIM))
        traj[:, 0, :] = z0
        for i in range(1, len(t)):
            traj[:, i, :] = odeint(traj[:, i - 1, :], t[i - 1], t[i])

        ctx.save_for_backward(t, traj, theta)
        ctx.odeint = odeint

        return traj

    @staticmethod
    def backward(ctx, a):
        t, traj, theta = ctx.saved_tensors
        ODE_module = ctx.module

        def aug_dynamics(s, t):
            # return dfdz dfdtheta dfdt
            z, a, theta, f = s
            a_aug = (z, t, theta)
            adfdz, adfdt, adfdtheta = torch.autograd.grad(f, a_aug, grad_outpus=a)
            return adfdz, adfdt, adfdtheta

        adj_traj = np.array((BATCH, TIME, DIM))

        for i in range(0, len(t), -1):
            z_i = traj[i]
            t_i = t[i]
            theta_i = theta
            f_i = ODE_module(z_i, t_i)

            a_i = a[i]

            s_i = [z_i, a_i, theta_i, f_i]

            adj_traj[:, i, :] = odeint.ODESolve(s_i, aug_dynamics, t_i, t[i + 1])

        adj_z = adj_traj[:, :, DIM : 2 * DIM]
        adj_t = adj_traj[:, :, 2 * DIM : 2 * DIM + 1]
        adj_theta = adj_traj[:, :, 2 * DIM + 1 :]

        return adj_z, adj_t, adj_theta.sum(2)


class NODE(torch.nn.Module):
    def __init__(self, odeint):
        super(NODE, self).__init__()

        self.module = odeint

    def forward(self, z0, t):
        theta = self.module.parameters
        z = Adjoint_ODE.apply(z0, t, theta, self.module)
        return z
