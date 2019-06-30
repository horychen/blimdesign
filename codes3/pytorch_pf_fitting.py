import pandas as pd
data = pd.read_csv('run#540_PF_points.txt')
# print(data)
# print(data['TRV'])
# print(data['eta'])
# print(data['OC'])
# quit()

x = [[x1, x2] for x1, x2 in zip(data['TRV'], data['eta'])]
y = data['OC'].values.tolist()
n = len(y) // 2

import torch
import torch.nn as nn
import torch.nn.functional as F

x_train, y_train, x_valid, y_valid = map( torch.tensor, (x[:n], y[:n], x[n:], x[n:]) )
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
# print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

import math

weights = torch.randn(2, 4) / math.sqrt(2) # Xavier_initialisation
weights.requires_grad_()
bias = torch.zeros(4, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)


from pylab import np, plt
x = torch.arange(-1,1,1e-5)
plt.plot(x.tolist(), list(map(log_softmax, x)))
plt.show()

bs = 10  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)


# def nll(input, target):
#     return -input[range(target.shape[0]), target].mean()
# loss_func = nll
loss_func = nn.MSELoss()

yb = y_train[0:bs]
print(loss_func(preds, yb))


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        # set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

