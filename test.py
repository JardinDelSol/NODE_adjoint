import torch
import matplotlib.pyplot as plt
import numpy as np

from ode import *
from adjoint_ode import *

if __name__ == "__main__":
    EPOCH = 100
    BATCH = 1
    TIME = 10
    DIM = 2

    model = NODE(odeint())
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

    # data = torch.randn((BATCH, DIM))
    data = torch.tensor((0, 0))

    target_time = torch.arange(10)
    y_dim = torch.sin(target_time)

    label = torch.stack((target_time, y_dim)).permute(1, 0).unsqueeze(0)

    mse = torch.nn.MSELoss()
    # mse = torch.nn.MSELoss(reduction="none")

    for i in range(EPOCH):
        print("Epoch :", i)
        result = model(data, target_time)
        loss = mse(result, label)
        print("loss: ", loss)
        loss.backward()
        optimizer.step()

