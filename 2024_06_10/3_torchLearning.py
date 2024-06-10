import torch
import torch.nn as nn

X = torch.tensor ([1, 2, 3, 4], dtype = torch.float32)
y = torch.tensor ([2, 4, 6, 8], dtype = torch.float32)

w = torch.tensor (0.0, dtype = torch.float32, requires_grad = True)

def forward (X):
    return w * X

print (f'Prediction before training : f(5) = {forward (5):.3f}')

alpha = 0.01
epochs = 100

loss = nn.MSELoss ()
optimizer = torch.optim.SGD ([w], lr = alpha)

for epoch in range (epochs):
    y_pred = forward (X)
    l = loss (y, y_pred)
    l.backward ()
    optimizer.step ()
    optimizer.zero_grad ()

    if epoch % 1 == 0:
        print (f'Epoch {epoch + 2} : w = {w:.3f}, loss = {l:.8f}')

print (f'Prediction after training : f(5) = {forward (5):.3f}')