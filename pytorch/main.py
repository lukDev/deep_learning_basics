import torch
import numpy as np
import matplotlib.pyplot as plot
from pytorch.network import Network
rg = np.random.default_rng(1)


def squared(x):
    return x**2


model = Network()


def evaluate():
    inputs = torch.tensor(np.array([rg.random() * 2. - 1. for _ in range(100)]).reshape((100, 1, 1))).float()
    targets = torch.tensor(np.array([squared(i) for i in inputs]).reshape((100, 1, 1))).float()
    with torch.no_grad():
        outputs = model(inputs)

    plot.scatter(inputs[:, 0, 0], targets[:, 0, 0])
    plot.scatter(inputs[:, 0, 0], outputs[:, 0, 0])
    plot.show()


evaluate()

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
loss_func = torch.nn.MSELoss()
for i in range(5000):
    inputs = torch.tensor(np.array([rg.random() * 2. - 1. for _ in range(100)]).reshape((100, 1, 1))).float()
    targets = torch.tensor(np.array([squared(i) for i in inputs]).reshape((100, 1, 1))).float()
    outputs = model(inputs)

    loss = loss_func(outputs, targets)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print(f"epoch: {i+1}\tloss: {np.average(loss.detach().numpy())}")

evaluate()