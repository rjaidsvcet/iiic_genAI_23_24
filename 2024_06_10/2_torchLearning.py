import torch

X = torch.tensor ([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor ([2, 4, 6, 8], dtype=torch.float32)

# w = 0.0
w = torch.tensor (0.0, dtype=torch.float32, requires_grad=True)

def forward (X):
    return w * X

def loss (y, y_pred):
    return ((y_pred-y)**2).mean ()

# def gradient (X, y, y_pred):
#     return np.dot (2*X, y_pred-y).mean()

print (f'Prediction before training : f(5) = {forward (5):.3f}')

epochs = 50
alpha = 0.01

for epoch in range (epochs):
    y_pred = forward (X)
    l = loss (y_pred, y)
    l.backward ()
    with torch.no_grad ():
        w -= alpha * w.grad
    # dw = gradient (X, y, y_pred)
    # w -= alpha * dw
    w.grad.zero_()

    if epoch % 1 == 0:
        print (f'Epoch {epoch + 2} : w = {w:.3f}, loss = {l:.8f}')


print (f'Prediction before training : f(5) = {forward (5):.3f}')
