import numpy as np

X = np.array ([1, 2, 3, 4], dtype=np.float32)
y = np.array ([2, 4, 6, 8], dtype=np.float32)

w = 0.0

def forward (X):
    return w * X

def loss (y, y_pred):
    return ((y_pred-y)**2).mean ()

def gradient (X, y, y_pred):
    return np.dot (2*X, y_pred-y).mean()

print (f'Prediction before training : f(5) = {forward (5):.3f}')

epochs = 20
alpha = 0.01

for epoch in range (epochs):
    y_pred = forward (X)
    l = loss (y_pred, y)
    dw = gradient (X, y, y_pred)
    w -= alpha * dw

    if epoch % 1 == 0:
        print (f'Epoch {epoch + 2} : w = {w:.3f}, loss = {l:.8f}')


print (f'Prediction before training : f(5) = {forward (5):.3f}')
