import torch
import numpy as np

# Hyper parameters
BATCH = 1
TIME = 10
DIM = 2

MIN_STEP = 0.1


def ODESolve(z0, f, t0, t1):
    z1 = z0
    while t0 <= t1:
        input = torch.cat((z1, t0.reshape(1, 1)), axis=1)
        z1 += MIN_STEP * f(input)
        t0 = t0 + MIN_STEP
    return z1


class odeint(torch.nn.Module):
    def __init__(self):
        super(odeint, self).__init__()
        self.model = torch.nn.Linear(DIM + 1, DIM)

    def forward(self, z0, t0, t1):
        return ODESolve(z0, self.model, t0, t1)

    # def adjoint_ODESolve(self, z0, f, t0, t1, a, theta, output):
    #     z1 = z0
    #     while t0 <= t1:
    #         z1 += MIN_STEP * f(z1, a, t0, theta, output)
    #         t0 += MIN_STEP
    #     return z1
